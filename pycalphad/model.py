"""
The model module provides support for using a Database to perform
calculations under specified conditions.
"""
import copy
import warnings
from symengine import exp, log, Abs, Add, And, Float, Mul, Piecewise, Pow, S, sin, StrictGreaterThan, Symbol, zoo, oo
from tinydb import where
import pycalphad.variables as v
from pycalphad.core.errors import DofError
from pycalphad.core.constants import MIN_SITE_FRACTION
from pycalphad.core.utils import unpack_components, get_pure_elements, wrap_symbol
import numpy as np
from collections import OrderedDict

# Maximum number of levels deep we check for symbols that are functions of
# other symbols
_MAX_PARAM_NESTING = 32


def _toop_filter(chemical_group_dict, symmetric_species, asymmetric_species):
    """
    Return a function ``f(m)`` that returns ``True`` if m is symmetric with
    the symmetric_species and asymmetric with the asymmetric_species.

    I.e. returns True if "j" is the asymmetric (Toop-like) species in the i-j-m ternary
    """
    def _f(species):
        if species == symmetric_species:
            return False
        elif species == asymmetric_species:
            return False
        elif chemical_group_dict[species] == chemical_group_dict[symmetric_species] and chemical_group_dict[species] != chemical_group_dict[asymmetric_species]:
            return True  # This chemical group should be mixed
        else:
            return False
    return _f


def _kohler_filter(chemical_group_dict, symmetric_species_1, symmetric_species_2):
    """
    Return a function ``f(m)`` that returns ``True`` if m is symmetric with
    the symmetric_species_1 and with symmetric_species_2.
    """
    def _f(species):
        if species == symmetric_species_1:
            return False
        elif species == symmetric_species_2:
            return False
        elif chemical_group_dict[species] == chemical_group_dict[symmetric_species_1] and chemical_group_dict[species] == chemical_group_dict[symmetric_species_2]:
            return True  # This chemical group should be mixed
        else:
            return False
    return _f

class ReferenceState():
    """
    Define the phase and any fixed state variables as a reference state for a component.

    Parameters
    ----------

    Attributes
    ----------
    fixed_statevars : dict
        Dictionary of {StateVariable: value} that will be fixed, e.g. {v.T: 298.15, v.P: 101325}
    phase_name : str
        Name of phase
    species : Species
        pycalphad Species variable

    """
    def __init__(self, species, reference_phase, fixed_statevars=None):
        """
        Parameters
        ----------
        species : str or Species
            Species to define the reference state for. Only pure elements supported.
        reference_phase : str
            Name of phase
        fixed_statevars : None, optional
            Dictionary of {StateVariable: value} that will be fixed, e.g. {v.T: 298.15, v.P: 101325}
            If None (the default), an empty dict will be created.

        """
        if isinstance(species, v.Species):
            self.species = species
        else:
            self.species = v.Species(species)
        self.phase_name = reference_phase
        self.fixed_statevars = fixed_statevars if fixed_statevars is not None else {}

    def __repr__(self):
        if len(self.fixed_statevars.keys()) > 0:
            s = "ReferenceState('{}', '{}', {})".format(self.species.name, self.phase_name, self.fixed_statevars)
        else:
            s = "ReferenceState('{}', '{}')".format(self.species.name, self.phase_name)
        return s


class Model(object):
    """
    Models use an abstract representation of the function
    for calculation of values under specified conditions.

    Parameters
    ----------
    dbe : Database
        Database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    phase_name : str
        Name of phase model to build.
    parameters : dict or list
        Optional dictionary of parameters to be substituted in the model.
        A list of parameters will cause those symbols to remain symbolic.
        This will overwrite parameters specified in the database

    Attributes
    ----------
    constituents : List[Set[Species]]
        A list of sublattices containing the sets of species on each sublattice.

    Methods
    -------
    None yet.

    Examples
    --------
    None yet.

    Notes
    -----
    The two sublattice ionic liquid model has several special cases compared to
    typical models within the compound energy formalism. A few key differences
    arise. First, variable site ratios (modulated by vacancy site fractions)
    are used to charge balance the phase. Second, endmembers for neutral
    species and interactions among only neutral species should be specified
    using only one sublattice (dropping the cation sublattice). For
    understanding the special cases used throughout this class, users are
    referred to:
    Sundman, "Modification of the two-sublattice model for liquids",
    Calphad 15(2) (1991) 109-119 https://doi.org/d3jppb


    """
    # We only use the contributions attribute in build_phase.
    # Users should not access it later since subclasses can override build_phase
    # and make self.models inconsistent with contributions.
    # Note that we include atomic ordering last since it uses self.models
    # to figure out its contribution.
    contributions = [('ref', 'reference_energy'), ('idmix', 'ideal_mixing_energy'),
                     ('xsmix', 'excess_mixing_energy'), ('mag', 'magnetic_energy'),
                     ('2st', 'twostate_energy'), ('ein', 'einstein_energy'),
                     ('ord', 'atomic_ordering_energy')]

    def __new__(cls, *args, **kwargs):
        target_cls = cls._dispatch_on(*args, **kwargs)
        instance = object.__new__(target_cls)
        return instance

    @classmethod
    def _dispatch_on(cls, dbe, comps, phase_name, parameters=None):
        phase = dbe.phases[phase_name.upper()]
        target_cls = cls
        if 'mqmqa' in phase.model_hints.keys():
            from pycalphad.models.model_mqmqa import ModelMQMQA
            target_cls = ModelMQMQA
        return target_cls

    def __getnewargs_ex__(self):
        return ((self._dbe, self.components, self.phase_name,), {'parameters': self._parameters_arg})

    def __init__(self, dbe, comps, phase_name, parameters=None):
        self._dbe = dbe
        self._endmember_reference_model = None
        self.components = set()
        self.constituents = []
        self.phase_name = phase_name.upper()
        phase = dbe.phases[self.phase_name]
        self.site_ratios = list(phase.sublattices)
        active_species = unpack_components(dbe, comps)
        for idx, sublattice in enumerate(phase.constituents):
            subl_comps = set(sublattice).intersection(active_species)
            self.components |= subl_comps
            # Support for variable site ratios in ionic liquid model
            if phase.model_hints.get('ionic_liquid_2SL', False):
                if idx == 0:
                    subl_idx = 1
                elif idx == 1:
                    subl_idx = 0
                else:
                    raise ValueError('Two-sublattice ionic liquid specified with more than two sublattices')
                self.site_ratios[subl_idx] = Add(*[v.SiteFraction(self.phase_name, idx, spec) * abs(spec.charge) for spec in subl_comps])
        if phase.model_hints.get('ionic_liquid_2SL', False):
            # Special treatment of "neutral" vacancies in 2SL ionic liquid
            # These are treated as having variable valence
            for idx, sublattice in enumerate(phase.constituents):
                subl_comps = set(sublattice).intersection(active_species)
                if v.Species('VA') in subl_comps:
                    if idx == 0:
                        subl_idx = 1
                    elif idx == 1:
                        subl_idx = 0
                    else:
                        raise ValueError('Two-sublattice ionic liquid specified with more than two sublattices')
                    self.site_ratios[subl_idx] += self.site_ratios[idx] * v.SiteFraction(self.phase_name, idx, v.Species('VA'))
        self.site_ratios = tuple(self.site_ratios)

        # Verify that this phase is still possible to build
        is_pure_VA = set()
        for sublattice in phase.constituents:
            sublattice_comps = set(sublattice).intersection(self.components)
            if len(sublattice_comps) == 0:
                # None of the components in a sublattice are active
                # We cannot build a model of this phase
                raise DofError(
                    '{0}: Sublattice {1} of {2} has no components in {3}' \
                    .format(self.phase_name, sublattice,
                            phase.constituents,
                            self.components))
            is_pure_VA.add(sum(set(map(lambda s : getattr(s, 'number_of_atoms'),sublattice_comps))))
            self.constituents.append(sublattice_comps)
        if sum(is_pure_VA) == 0:
            #The only possible component in a sublattice is vacancy
            #We cannot build a model of this phase
            raise DofError(
                '{0}: Sublattices of {1} contains only VA (VACUUM) constituents' \
                .format(self.phase_name, phase.constituents))
        self.components = sorted(self.components)
        desired_active_pure_elements = [list(x.constituents.keys()) for x in self.components]
        desired_active_pure_elements = [el.upper() for constituents in desired_active_pure_elements
                                        for el in constituents]
        self.pure_elements = sorted(set(desired_active_pure_elements))
        self.nonvacant_elements = [x for x in self.pure_elements if x != 'VA']

        # Convert string symbol names to Symbol objects
        # This makes xreplace work with the symbols dict
        symbols = {Symbol(s): val for s, val in dbe.symbols.items()}

        if parameters is not None:
            self._parameters_arg = parameters
            if isinstance(parameters, dict):
                symbols.update([(wrap_symbol(s), val) for s, val in parameters.items()])
            else:
                # Lists of symbols that should remain symbolic
                for s in parameters:
                    symbols.pop(wrap_symbol(s))
        else:
            self._parameters_arg = None

        self._symbols = {wrap_symbol(key): value for key, value in symbols.items()}

        self.models = OrderedDict()
        self.build_phase(dbe)

        for name, value in self.models.items():
            # XXX: xreplace hack because SymEngine seems to let Symbols slip in somehow
            self.models[name] = self.symbol_replace(value, symbols).xreplace(v.supported_variables_in_databases)

        self.site_fractions = sorted([x for x in self.variables if isinstance(x, v.SiteFraction)], key=str)
        self.state_variables = sorted([x for x in self.variables if not isinstance(x, v.SiteFraction)], key=str)

    @staticmethod
    def symbol_replace(obj, symbols):
        """
        Substitute values of symbols into 'obj'.

        Parameters
        ----------
        obj : SymEngine object
        symbols : dict mapping symengine.Symbol to SymEngine object

        Returns
        -------
        SymEngine object
        """
        try:
            # Need to do more substitutions to catch symbols that are functions
            # of other symbols
            for iteration in range(_MAX_PARAM_NESTING):
                obj = obj.xreplace(symbols)
                undefs = [x for x in obj.free_symbols if not isinstance(x, v.StateVariable)]
                if len(undefs) == 0:
                    break
        except AttributeError:
            # Can't use xreplace on a float
            pass
        return obj

    def __eq__(self, other):
        if self is other:
            return True
        elif type(self) != type(other):
            return False
        else:
            return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(repr(self))

    def moles(self, species, per_formula_unit=False):
        "Number of moles of species or elements."
        species = v.Species(species)
        is_pure_element = (len(species.constituents.keys()) == 1 and
                           list(species.constituents.keys())[0] == species.name)
        result = S.Zero
        normalization = S.Zero
        if is_pure_element:
            element = list(species.constituents.keys())[0]
            for idx, sublattice in enumerate(self.constituents):
                active = set(sublattice).intersection(self.components)
                result += self.site_ratios[idx] * \
                    sum(int(spec.number_of_atoms > 0) * spec.constituents.get(element, 0) * v.SiteFraction(self.phase_name, idx, spec)
                        for spec in active)
                normalization += self.site_ratios[idx] * \
                    sum(spec.number_of_atoms * v.SiteFraction(self.phase_name, idx, spec)
                        for spec in active)
        else:
            for idx, sublattice in enumerate(self.constituents):
                active = set(sublattice).intersection({species})
                if len(active) == 0:
                    continue
                result += self.site_ratios[idx] * sum(v.SiteFraction(self.phase_name, idx, spec) for spec in active)
                normalization += self.site_ratios[idx] * \
                    sum(int(spec.number_of_atoms > 0) * v.SiteFraction(self.phase_name, idx, spec)
                        for spec in active)
        if not per_formula_unit:
            return result / normalization
        else:
            return result

    @property
    def ast(self):
        "Return the full abstract syntax tree of the model."
        return Add(*list(self.models.values()))

    @property
    def variables(self):
        "Return state variables in the model."
        return sorted([x for x in self.ast.free_symbols if isinstance(x, v.StateVariable)], key=str)

    @property
    def degree_of_ordering(self):
        result = S.Zero
        site_ratio_normalization = S.Zero
        # Calculate normalization factor
        for idx, sublattice in enumerate(self.constituents):
            active = set(sublattice).intersection(self.components)
            subl_content = sum(int(spec.number_of_atoms > 0) * v.SiteFraction(self.phase_name, idx, spec) for spec in active)
            site_ratio_normalization += self.site_ratios[idx] * subl_content

        site_ratios = [c/site_ratio_normalization for c in self.site_ratios]
        for comp in self.components:
            if comp.number_of_atoms == 0:
                continue
            comp_result = S.Zero
            for idx, sublattice in enumerate(self.constituents):
                active = set(sublattice).intersection(set(self.components))
                if comp in active:
                    comp_result += site_ratios[idx] * Abs(v.SiteFraction(self.phase_name, idx, comp) - self.moles(comp)) / self.moles(comp)
            result += comp_result
        return result / sum(int(spec.number_of_atoms > 0) for spec in self.components)
    DOO = degree_of_ordering

    # Can be defined as a list of pre-computed first derivatives
    gradient = None

    # Note: In order-disorder phases, TC will always be the *disordered* value of TC
    curie_temperature = TC = S.Zero
    beta = BMAG = S.Zero
    neel_temperature = NT = S.Zero

    #pylint: disable=C0103
    # These are standard abbreviations from Thermo-Calc for these quantities
    energy = GM = property(lambda self: self.ast)
    formulaenergy = G = property(lambda self: self.ast * self._site_ratio_normalization)
    entropy = SM = property(lambda self: -self.GM.diff(v.T))
    enthalpy = HM = property(lambda self: self.GM - v.T*self.GM.diff(v.T))
    heat_capacity = CPM = property(lambda self: -v.T*self.GM.diff(v.T, v.T))
    #pylint: enable=C0103
    mixing_energy = GM_MIX = property(lambda self: self.GM - self.endmember_reference_model.GM)
    mixing_enthalpy = HM_MIX = property(lambda self: self.GM_MIX - v.T*self.GM_MIX.diff(v.T))
    mixing_entropy = SM_MIX = property(lambda self: -self.GM_MIX.diff(v.T))
    mixing_heat_capacity = CPM_MIX = property(lambda self: -v.T*self.GM_MIX.diff(v.T, v.T))

    @property
    def endmember_reference_model(self):
        """
        Return a Model containing only energy contributions from endmembers.

        Returns
        -------
        Model

        Notes
        -----
        The endmember_reference_model is used for ``_MIX`` properties of Model objects.
        It is defined such that subtracting it from the model will set the energy of the
        endmembers to zero. The endmember_reference_model AST can be modified in the
        same way as any Model.

        Partitioned models have energetic contributions from the ordered compound
        energies/interactions and the disordered compound energies/interactions.
        The endmembers to choose as the reference is ambiguous. If the current model has
        an ordered energy as part of a partitioned model, then the model energy
        contributions are set to ``nan``.

        The endmember reference model is built lazily and stored for later re-use
        because it needs to copy the Database and instantiate a new Model.
        """
        if self._endmember_reference_model is None:
            endmember_only_dbe = copy.deepcopy(self._dbe)
            endmember_only_dbe._parameters.remove(where('constituent_array').test(self._interaction_test))
            mod_endmember_only = self.__class__(endmember_only_dbe, self.components, self.phase_name, parameters=self._parameters_arg)
            # Ideal mixing contributions are always generated, so we need to set the
            # contribution of the endmember reference model to zero to preserve ideal
            # mixing in this model.
            mod_endmember_only.models['idmix'] = 0
            if self.models.get('ord', S.Zero) != S.Zero:
                warnings.warn(
                    f"{self.phase_name} is a partitioned model with an ordering energy "
                    "contribution. The choice of endmembers for the endmember "
                    "reference model used by `_MIX` properties is ambiguous for "
                    "partitioned models. The `Model.set_reference_state` method is a "
                    "better choice for computing mixing energy. See "
                    "https://pycalphad.org/docs/latest/examples/ReferenceStateExamples.html "
                    "for an example."
                )
                for k in mod_endmember_only.models.keys():
                    mod_endmember_only.models[k] = float('nan')
            self._endmember_reference_model = mod_endmember_only
        return self._endmember_reference_model

    def get_internal_constraints(self):
        constraints = []
        # Site fraction balance
        for idx, sublattice in enumerate(self.constituents):
            constraints.append(sum(v.SiteFraction(self.phase_name, idx, spec) for spec in sublattice) - 1)
        # Charge balance for all phases that are charged
        has_charge = len({sp for sp in self.components if sp.charge != 0}) > 0
        constant_site_ratios = True
        # The only implementation with variable site ratios is the two-sublattice ionic liquid.
        # This check is convenient for detecting 2SL ionic liquids without keeping other state.
        # Because 2SL ionic liquids charge balance 'automatically', we do not need to enforce charge balance.
        for sr in self.site_ratios:
            try:
                float(sr)
            except (TypeError, RuntimeError):
                constant_site_ratios = False
        # For all other cases where charge is present, we do need to add charge balance.
        if constant_site_ratios and has_charge:
            total_charge = 0
            for idx, (sublattice, site_ratio) in enumerate(zip(self.constituents, self.site_ratios)):
                total_charge += sum(v.SiteFraction(self.phase_name, idx, spec) * spec.charge * site_ratio
                                    for spec in sublattice)
            constraints.append(total_charge)
        return constraints


    def build_phase(self, dbe):
        """
        Generate the symbolic form of all the contributions to this phase.

        Parameters
        ----------
        dbe : Database
        """
        contrib_vals = list(OrderedDict(self.__class__.contributions).values())
        if 'atomic_ordering_energy' in contrib_vals:
            if contrib_vals.index('atomic_ordering_energy') != (len(contrib_vals) - 1):
                # Check for a common mistake in custom models
                # Users that need to override this behavior should override build_phase
                raise ValueError('\'atomic_ordering_energy\' must be the final contribution')
        self.models.clear()
        for key, value in self.__class__.contributions:
            self.models[key] = S(getattr(self, value)(dbe))

    def _array_validity(self, constituent_array):
        """
        Return True if the constituent_array contains only active species of the current Model instance.
        """
        if len(constituent_array) != len(self.constituents):
            # Allow an exception for the ionic liquid model, where neutral
            # species can be specified in the anion sublattice without any
            # species in the cation sublattice.
            ionic_liquid_2SL = self._dbe.phases[self.phase_name].model_hints.get('ionic_liquid_2SL', False)
            if ionic_liquid_2SL and len(constituent_array) == 1:
                param_sublattice = constituent_array[0]
                model_anion_sublattice = self.constituents[1]
                if (set(param_sublattice).issubset(model_anion_sublattice) or (param_sublattice[0] == v.Species('*'))):
                    return True
            return False
        for param_sublattice, model_sublattice in zip(constituent_array, self.constituents):
            if not (set(param_sublattice).issubset(model_sublattice) or (param_sublattice[0] == v.Species('*'))):
                return False
        return True

    def _purity_test(self, constituent_array):
        """
        Return True if the constituent_array is valid and has exactly one
        species in every sublattice.
        """
        if not self._array_validity(constituent_array):
            return False
        return not any(len(sublattice) != 1 for sublattice in constituent_array)

    def _interaction_test(self, constituent_array):
        """
        Return True if the constituent_array is valid and has more than one
        species in at least one sublattice.
        """
        if not self._array_validity(constituent_array):
            return False
        return any([len(sublattice) > 1 for sublattice in constituent_array])

    @property
    def _site_ratio_normalization(self):
        """
        Calculates the normalization factor based on the number of sites
        in each sublattice.
        """
        site_ratio_normalization = S.Zero
        # Calculate normalization factor
        for idx, sublattice in enumerate(self.constituents):
            active = set(sublattice).intersection(self.components)
            subl_content = sum(spec.number_of_atoms * v.SiteFraction(self.phase_name, idx, spec) for spec in active)
            site_ratio_normalization += self.site_ratios[idx] * subl_content
        return site_ratio_normalization

    @staticmethod
    def _Muggianu_correction_dict(comps): #pylint: disable=C0103
        """
        Replace y_i -> y_i + (1 - sum(y involved in parameter)) / m,
        where m is the arity of the interaction parameter.
        Returns a dict converting the list of Symbols (comps) to this.
        m is assumed equal to the length of comps.

        When incorporating binary, ternary or n-ary interaction parameters
        into systems with more than n components, the sum of site fractions
        involved in the interaction parameter may no longer be unity. This
        breaks the symmetry of the parameter. The solution suggested by
        Muggianu, 1975, is to renormalize the site fractions by replacing them
        with a term that will sum to unity even in higher-order systems.
        There are other solutions that involve retaining the asymmetry for
        physical reasons, but this solution works well for components that
        are physically similar.

        This procedure is based on an analysis by Hillert, 1980,
        published in the Calphad journal.
        """
        arity = len(comps)
        return_dict = {}
        correction_term = (S.One - Add(*comps)) / arity
        for comp in comps:
            return_dict[comp] = comp + correction_term
        return return_dict

    def _Xi_ij(self, phase, sublattice_index, i, j):
        # Species of the same chemical group may use a Kohler-type or
        # Muggianu-type approximation (the choice does not affect this function).
        # Species of different chemical groups use a Toop-type approximation.
        toop_filter = _toop_filter(phase.model_hints['chemical_groups'], i, j)
        toop_correction = S.Zero
        active_subl_comps = phase.constituents[sublattice_index].intersection(self.components)
        for k in filter(toop_filter, active_subl_comps):
            toop_correction += v.Y(self.phase_name, sublattice_index, k)
        return v.Y(self.phase_name, sublattice_index, i) + toop_correction

    def _sigma_ij(self, phase, sublattice_index, i, j):
        # Compute the correction for Kohler-type approximation between species i and j
        # We don't yet have a way to flag whether the ij interaction, if the
        # species are of the same chemical group (they are "symmetric", by
        # Pelton) uses a Kohler-type or Muggianu-type approximation. Since
        # there exists the _Muggianu_correction_dict already, we assume that
        # callers of this function are only interested in a Kohler-type
        # approximation for species of the same type. Species of different
        # chemical groups use a Toop-type approximation.
        kohler_filter = _kohler_filter(phase.model_hints['chemical_groups'], i, j)
        kohler_correction = S.Zero
        active_subl_comps = phase.constituents[sublattice_index].intersection(self.components)
        for k in filter(kohler_filter, active_subl_comps):
            kohler_correction += v.Y(self.phase_name, sublattice_index, k)
        return 1 -  kohler_correction

    def _alpha_ij_Q(self, phase, sublattice_index, i, j, q_ij, exp_i, exp_j):
        Xi_ij = self._Xi_ij(phase, sublattice_index, i, j)
        Xi_ji = self._Xi_ij(phase, sublattice_index, j, i)
        sigma_ij = self._sigma_ij(phase, sublattice_index, i, j) # same for both
        i_term = (1 + (Xi_ij - Xi_ji) / sigma_ij) / 2
        j_term = (1 + (Xi_ji - Xi_ij) / sigma_ij) / 2
        return q_ij * i_term**exp_i * j_term**exp_j

    def _alpha_ij_L(self, phase, sublattice_index, i, j, L_ij, parameter_order):
        # Use the parameter-sorted order for i and j, noting that L_{ij} != L_{ji} for odd-ordered terms.
        Xi_ij = self._Xi_ij(phase, sublattice_index, i, j)
        Xi_ji = self._Xi_ij(phase, sublattice_index, j, i)
        sigma_ij = self._sigma_ij(phase, sublattice_index, i, j)
        mixing_term = (Xi_ij - Xi_ji) / sigma_ij
        return L_ij * mixing_term**parameter_order

    def kohler_toop_excess_sum(self, dbe):
        phase = dbe.phases[self.phase_name]
        param_query = (
            (where("phase_name") == phase.name) &
            (where("parameter_type") == "QKT") &
            (where('constituent_array').test(self._array_validity))
        )
        param_search = dbe.search

        params = param_search(param_query)
        kohler_toop_xs = S.Zero
        for param in params:
            mixing_term = S.One
            for subl_idx, subl_comps in enumerate(param["constituent_array"]):
                mixing_term *= Mul(*[v.SiteFraction(phase.name, subl_idx, comp) for comp in subl_comps])
                if len(subl_comps) == 2:
                    # Note: The structure of exponents (a `List[int]`) currently assumes only one sublattice has mixing
                    assert len(subl_comps) == len(param["exponents"])
                    i, j = subl_comps
                    p, q = param["exponents"]
                    alpha_ij_Q = self._alpha_ij_Q(phase, subl_idx, i, j, param["parameter"], p, q)
                    mixing_term *= alpha_ij_Q
                elif len(subl_comps) == 3:
                    # Pelton's 2001 paper lays out several different flavors of
                    # ternary parameters, but the simple flavor of ternary
                    # parameters (Pelton 2001 Eq. 17) seems to be used in the
                    # Quasichemical Kohler-Toop model implemented in DAT files.
                    # Note: The structure of exponents (a `List[int]`) currently assumes only one sublattice has mixing
                    exponents = param["exponents"]
                    assert len(subl_comps) == len(exponents)
                    mixing_term *= Mul(*[v.SiteFraction(phase.name, subl_idx, comp)**expn for comp, expn in zip(subl_comps, exponents)])
                    # Note, the following normalization is not in the paper
                    # either, but the equation in the paper clearly states that
                    # this type of "simple" ternary is discouraged and doesn't
                    # talk about how to extrapolate into multi-component.
                    mixing_term /= Add(*[v.SiteFraction(phase.name, subl_idx, comp) for comp in subl_comps])**(sum(exponents))
                    mixing_term *= param["parameter"]
                else:
                    raise ValueError(f"Unsupported number of components are mixing, got {len(subl_comps)} ({subl_comps}), expected 2 or 3.")
            kohler_toop_xs += mixing_term
        return kohler_toop_xs


    def redlich_kister_sum(self, phase, param_search, param_query):
        """
        Construct parameter in Redlich-Kister polynomial basis, using
        the Muggianu ternary parameter extension.
        """
        rk_terms = []

        # search for desired parameters
        params = param_search(param_query)
        for param in params:
            # iterate over every sublattice
            mixing_term = S.One
            for subl_index, comps in enumerate(param['constituent_array']):
                comp_symbols = None
                # convert strings to symbols
                if comps[0] == v.Species('*'):
                    # Handle wildcards in constituent array
                    comp_symbols = \
                        [
                            v.SiteFraction(phase.name, subl_index, comp)
                            for comp in sorted(set(phase.constituents[subl_index])\
                                .intersection(self.components))
                        ]
                    mixing_term *= Add(*comp_symbols)
                else:
                    if (
                        phase.model_hints.get('ionic_liquid_2SL', False) and  # This is an ionic 2SL
                        len(param['constituent_array']) == 1 and  # There's only one sublattice
                        all(const.charge == 0 for const in param['constituent_array'][0])  # All constituents are neutral
                    ):
                        # The constituent array is all neutral anion species in what would be the
                        # second sublattice. TDB syntax allows for specifying neutral species with
                        # one sublattice model. Set the sublattice index to 1 for the purpose of
                        # site fractions.
                        subl_index = 1
                    comp_symbols = \
                        [
                            v.SiteFraction(phase.name, subl_index, comp)
                            for comp in comps
                        ]
                    if phase.model_hints.get('ionic_liquid_2SL', False):  # This is an ionic 2SL
                        # We need to special case sorting for this model, because the constituents
                        # should not be alphabetically sorted. The model should be (C)(A, Va, B)
                        # for cations (C), anions (A), vacancies (Va) and neutrals (B). Thus the
                        # second sublattice should be sorted by species with charge, then by
                        # vacancies, if present, then by neutrals. Hint: in Thermo-Calc, using
                        # `set-start-constitution` for a phase will prompt you to enter site
                        # fractions for species in the order they are sorted internally within
                        # Thermo-Calc. This can be used to verify sorting behavior.

                        # Assume that the constituent array is already in sorted order
                        # alphabetically, so we need to rearrange the species first by charged
                        # species, then VA, then netural species. Since the cation sublattice
                        # should only have charged species by definition, this is equivalent to
                        # a no-op for the first sublattice.
                        charged_symbols = [sitefrac for sitefrac in comp_symbols if sitefrac.species.charge != 0 and sitefrac.species.number_of_atoms > 0]
                        va_symbols = [sitefrac for sitefrac in comp_symbols if sitefrac.species == v.Species('VA')]
                        neutral_symbols = [sitefrac for sitefrac in comp_symbols if sitefrac.species.charge == 0 and sitefrac.species.number_of_atoms > 0]
                        comp_symbols = charged_symbols + va_symbols + neutral_symbols

                    mixing_term *= Mul(*comp_symbols)
                # is this a higher-order interaction parameter?
                if len(comps) == 2 and param['parameter_order'] > 0:
                    # interacting sublattice, add the interaction polynomial
                    mixing_term *= Pow(comp_symbols[0] - \
                        comp_symbols[1], param['parameter_order'])
                if len(comps) == 3:
                    # 'parameter_order' is an index to a variable when
                    # we are in the ternary interaction parameter case

                    # NOTE: The commercial software packages seem to have
                    # a "feature" where, if only the zeroth
                    # parameter_order term of a ternary parameter is specified,
                    # the other two terms are automatically generated in order
                    # to make the parameter symmetric.
                    # In other words, specifying only this parameter:
                    # PARAMETER G(FCC_A1,AL,CR,NI;0) 298.15  +30300; 6000 N !
                    # Actually implies:
                    # PARAMETER G(FCC_A1,AL,CR,NI;0) 298.15  +30300; 6000 N !
                    # PARAMETER G(FCC_A1,AL,CR,NI;1) 298.15  +30300; 6000 N !
                    # PARAMETER G(FCC_A1,AL,CR,NI;2) 298.15  +30300; 6000 N !
                    #
                    # If either 1 or 2 is specified, no implicit parameters are
                    # generated.
                    # We need to handle this case.
                    if param['parameter_order'] == 0:
                        # are _any_ of the other parameter_orders specified?
                        ternary_param_query = (
                            (where('phase_name') == param['phase_name']) & \
                            (where('parameter_type') == \
                                param['parameter_type']) & \
                            (where('constituent_array') == \
                                param['constituent_array'])
                        )
                        other_tern_params = param_search(ternary_param_query)
                        if len(other_tern_params) == 1 and \
                            other_tern_params[0] == param:
                            # only the current parameter is specified
                            # We need to generate the other two parameters.
                            order_one = copy.copy(param)
                            order_one['parameter_order'] = 1
                            order_two = copy.copy(param)
                            order_two['parameter_order'] = 2
                            # Add these parameters to our iteration.
                            params.extend((order_one, order_two))
                    # Include variable indicated by parameter order index
                    # Perform Muggianu adjustment to site fractions
                    mixing_term *= comp_symbols[param['parameter_order']].subs(
                        self._Muggianu_correction_dict(comp_symbols))
            if phase.model_hints.get('ionic_liquid_2SL', False):
                # Special normalization rules for parameters apply under this model
                # If there are no anions present in the anion sublattice (only VA and neutral
                # species), then the energy has an additional Q*y(VA) term
                anions_present = any([m.species.charge < 0 for m in mixing_term.free_symbols])
                if not anions_present:
                    pair_rule = {}
                    # Cation site fractions must always appear with vacancy site fractions
                    va_subls = [(v.Species('VA') in phase.constituents[idx]) for idx in range(len(phase.constituents))]
                    # The last index that contains a vacancy
                    va_subl_idx = (len(phase.constituents) - 1) - va_subls[::-1].index(True)
                    va_present = any((v.Species('VA') in c) for c in param['constituent_array'])
                    if va_present and (max(len(c) for c in param['constituent_array']) == 1):
                        # No need to apply pair rule for VA-containing endmember
                        pass
                    elif va_subl_idx > -1:
                        for sym in mixing_term.free_symbols:
                            if sym.species.charge > 0:
                                pair_rule[sym] = sym * v.SiteFraction(sym.phase_name, va_subl_idx, v.Species('VA'))
                    mixing_term = mixing_term.xreplace(pair_rule)
                    # This parameter is normalized differently due to the variable charge valence of vacancies
                    mixing_term *= self.site_ratios[va_subl_idx]
            param_val = param['parameter']
            if isinstance(param_val, Piecewise):
                # Eliminate redundant Piecewise and extrapolate beyond temperature limits
                filtered_args = [expr for expr, cond in zip(*[iter(param_val.args)]*2) if not ((cond == S.true) and (expr == S.Zero))]
                if len(filtered_args) == 1:
                    param_val = filtered_args[0]
            rk_terms.append(mixing_term * param_val)
        return Add(*rk_terms)

    def reference_energy(self, dbe):
        """
        Returns the weighted average of the endmember energies
        in symbolic form.
        """
        pure_param_query = (
            (where('phase_name') == self.phase_name) & \
            (where('parameter_order') == 0) & \
            (where('parameter_type') == "G") & \
            (where('constituent_array').test(self._purity_test))
        )
        phase = dbe.phases[self.phase_name]
        param_search = dbe.search
        pure_energy_term = self.redlich_kister_sum(phase, param_search,
                                                   pure_param_query)
        return pure_energy_term / self._site_ratio_normalization

    def ideal_mixing_energy(self, dbe):
        #pylint: disable=W0613
        """
        Returns the ideal mixing energy in symbolic form.
        """
        phase = dbe.phases[self.phase_name]
        site_ratios = self.site_ratios
        ideal_mixing_term = S.Zero
        sitefrac_limit = Float(MIN_SITE_FRACTION/10.)
        for subl_index, sublattice in enumerate(phase.constituents):
            active_comps = set(sublattice).intersection(self.components)
            ratio = site_ratios[subl_index]
            for comp in active_comps:
                sitefrac = \
                    v.SiteFraction(phase.name, subl_index, comp)
                # We lose some precision here, but this makes the limit behave nicely
                # We're okay until fractions of about 1e-12 (platform-dependent)
                mixing_term = Piecewise((sitefrac*log(sitefrac),
                                         StrictGreaterThan(sitefrac, sitefrac_limit)), (0, True),
                                        )
                ideal_mixing_term += (mixing_term*ratio)
        ideal_mixing_term *= (v.R * v.T)
        return ideal_mixing_term / self._site_ratio_normalization

    def excess_mixing_energy(self, dbe):
        """
        Build the binary, ternary and higher order interaction term
        Here we use Redlich-Kister polynomial basis by default
        Here we use the Muggianu ternary extension by default
        Replace y_i -> y_i + (1 - sum(y involved in parameter)) / m,
        where m is the arity of the interaction parameter
        """
        phase = dbe.phases[self.phase_name]
        param_search = dbe.search
        param_query = (
            (where('phase_name') == self.phase_name) & \
                ((where('parameter_type') == 'G') |
                 (where('parameter_type') == 'L')) & \
                (where('constituent_array').test(self._interaction_test))
            )
        excess_term = self.redlich_kister_sum(phase, param_search, param_query)
        excess_term += self.kohler_toop_excess_sum(dbe)
        return excess_term / self._site_ratio_normalization

    def magnetic_energy(self, dbe):
        #pylint: disable=C0103, R0914
        """
        Return the energy from magnetic ordering in symbolic form.
        The implemented model is the Inden-Hillert-Jarl formulation.
        The approach follows from the background of W. Xiong et al, Calphad, 2012.
        """
        phase = dbe.phases[self.phase_name]
        param_search = dbe.search
        self.TC = self.curie_temperature = S.Zero
        self.BMAG = self.beta = S.Zero
        if 'ihj_magnetic_structure_factor' not in phase.model_hints:
            return S.Zero
        if 'ihj_magnetic_afm_factor' not in phase.model_hints:
            return S.Zero

        site_ratio_normalization = self._site_ratio_normalization
        # define basic variables
        afm_factor = phase.model_hints['ihj_magnetic_afm_factor']

        if afm_factor == 0:
            # Apply improved magnetic model which does not use AFM / Weiss factor
            return self.xiong_magnetic_energy(dbe)

        bm_param_query = (
            (where('phase_name') == phase.name) & \
            (where('parameter_type') == 'BMAGN') & \
            (where('constituent_array').test(self._array_validity))
        )
        tc_param_query = (
            (where('phase_name') == phase.name) & \
            (where('parameter_type') == 'TC') & \
            (where('constituent_array').test(self._array_validity))
        )

        mean_magnetic_moment = \
            self.redlich_kister_sum(phase, param_search, bm_param_query)
        beta = mean_magnetic_moment / Piecewise(
            (afm_factor, mean_magnetic_moment <= 0),
            (1., True)
            )
        self.BMAG = self.beta = self.symbol_replace(beta, self._symbols)

        curie_temp = \
            self.redlich_kister_sum(phase, param_search, tc_param_query)
        tc = curie_temp / Piecewise(
            (afm_factor, curie_temp <= 0),
            (1., True)
            )
        self.TC = self.curie_temperature = self.symbol_replace(tc, self._symbols)

        # Used to prevent singularity
        tau_positive_tc = v.T / (curie_temp + 1e-9)
        tau_negative_tc = v.T / ((curie_temp/afm_factor) + 1e-9)

        # define model parameters
        p = phase.model_hints['ihj_magnetic_structure_factor']
        A = 518/1125 + (11692/15975)*(1/p - 1)
        # factor when tau < 1 and tc < 0
        sub_tau_neg_tc = 1 - (1/A) * ((79/(140*p))*(tau_negative_tc**(-1)) + (474/497)*(1/p - 1) \
            * ((tau_negative_tc**3)/6 + (tau_negative_tc**9)/135 + (tau_negative_tc**15)/600)
                              )
        # factor when tau < 1 and tc > 0
        sub_tau_pos_tc = 1 - (1/A) * ((79/(140*p))*(tau_positive_tc**(-1)) + (474/497)*(1/p - 1) \
            * ((tau_positive_tc**3)/6 + (tau_positive_tc**9)/135 + (tau_positive_tc**15)/600)
                              )
        # factor when tau >= 1 and tc > 0
        super_tau_pos_tc = -(1/A) * ((tau_positive_tc**-5)/10 + (tau_positive_tc**-15)/315 + (tau_positive_tc**-25)/1500)
        # factor when tau >= 1 and tc < 0
        super_tau_neg_tc = -(1/A) * ((tau_negative_tc**-5)/10 + (tau_negative_tc**-15)/315 + (tau_negative_tc**-25)/1500)

        # This is an optimization to reduce the complexity of the compile-time expression
        expr_cond_pairs = [(sub_tau_neg_tc, curie_temp/afm_factor > v.T),
                           (sub_tau_pos_tc, curie_temp > v.T),
                           (super_tau_pos_tc, And(curie_temp < v.T, curie_temp > 0)),
                           (super_tau_neg_tc, And(curie_temp/afm_factor < v.T, curie_temp < 0)),
                           (0, True)
                           ]
        g_term = Piecewise(*expr_cond_pairs)

        return v.R * v.T * log(beta+1) * \
            g_term / site_ratio_normalization

    def xiong_magnetic_energy(self, dbe):
        """
        Return the energy from magnetic ordering in symbolic form.
        The approach follows W. Xiong et al, Calphad, 2012.
        """
        phase = dbe.phases[self.phase_name]
        param_search = dbe.search
        self.TC = self.curie_temperature = S.Zero
        if 'ihj_magnetic_structure_factor' not in phase.model_hints:
            return S.Zero
        if 'ihj_magnetic_afm_factor' not in phase.model_hints:
            return S.Zero

        site_ratio_normalization = self._site_ratio_normalization
        # define basic variables
        afm_factor = phase.model_hints['ihj_magnetic_afm_factor']

        if afm_factor != 0:
            raise ValueError('Xiong model called with nonzero AFM / Weiss factor')

        nt_param_query = (
            (where('phase_name') == phase.name) & \
            (where('parameter_type') == 'NT') & \
            (where('constituent_array').test(self._array_validity))
        )

        bm_param_query = (
            (where('phase_name') == phase.name) & \
            (where('parameter_type') == 'BMAGN') & \
            (where('constituent_array').test(self._array_validity))
        )
        tc_param_query = (
            (where('phase_name') == phase.name) & \
            (where('parameter_type') == 'TC') & \
            (where('constituent_array').test(self._array_validity))
        )

        mean_magnetic_moment = \
            self.redlich_kister_sum(phase, param_search, bm_param_query)
        beta = mean_magnetic_moment

        curie_temp = \
            self.redlich_kister_sum(phase, param_search, tc_param_query)
        neel_temp = \
            self.redlich_kister_sum(phase, param_search, nt_param_query)

        self.TC = self.curie_temperature = self.symbol_replace(curie_temp, self._symbols)
        self.NT = self.neel_temperature = self.symbol_replace(neel_temp, self._symbols)
        self.BMAG = self.beta = self.symbol_replace(beta, self._symbols)

        tau_curie = v.T / curie_temp
        tau_curie = tau_curie.xreplace({zoo: 1.0e10})
        tau_neel = v.T / neel_temp
        tau_neel = tau_neel.xreplace({zoo: 1.0e10})

        # define model parameters
        p = phase.model_hints['ihj_magnetic_structure_factor']
        D = 0.33471979 + 0.49649686*(1/p - 1)
        sub_tau_curie = 1 - (1/D) * ((0.38438376/p)*(tau_curie**(-1)) + 0.63570895*(1/p - 1) \
            * ((tau_curie**3)/6 + (tau_curie**9)/135 + (tau_curie**15)/600) + (tau_curie**21)/1617
                              )
        sub_tau_neel = 1 - (1/D) * ((0.38438376/p)*(tau_neel**(-1)) + 0.63570895*(1/p - 1) \
            * ((tau_neel**3)/6 + (tau_neel**9)/135 + (tau_neel**15)/600) + (tau_neel**21)/1617
                              )
        super_tau_curie = -(1/D) * ((tau_curie**-7)/21 + (tau_curie**-21)/630 + (tau_curie**-35)/2975 + (tau_curie**-49)/8232)
        super_tau_neel = -(1/D) * ((tau_neel**-7)/21 + (tau_neel**-21)/630 + (tau_neel**-35)/2975 + (tau_neel**-49)/8232)

        expr_cond_pairs_curie = [(0, tau_curie <= 0),
                                 (super_tau_curie, tau_curie > 1),
                                 (sub_tau_curie, True)
                                ]
        expr_cond_pairs_neel = [(0, tau_neel <= 0),
                                (super_tau_neel, tau_neel > 1),
                                (sub_tau_neel, True)
                               ]
        g_term = Piecewise(*expr_cond_pairs_curie) + Piecewise(*expr_cond_pairs_neel)

        return v.R * v.T * log(beta+1) * \
            g_term / site_ratio_normalization

    def twostate_energy(self, dbe):
        """
        Return the energy from liquid-amorphous two-state model.
        """
        phase = dbe.phases[self.phase_name]
        param_search = dbe.search
        site_ratio_normalization = self._site_ratio_normalization
        gd_param_query = (
            (where('phase_name') == phase.name) & \
            (where('parameter_type') == 'GD') & \
            (where('constituent_array').test(self._array_validity))
        )
        gd = self.redlich_kister_sum(phase, param_search, gd_param_query)
        if gd == S.Zero:
            return S.Zero
        return -v.R * v.T * log(1 + exp(-gd / (v.R * v.T))) / site_ratio_normalization

    def einstein_energy(self, dbe):
        """
        Return the energy based on the Einstein model.
        Note that THETA parameters are actually LN(THETA).
        All Redlich-Kister summation is done in log-space,
        then exp() is called on the result.
        """
        phase = dbe.phases[self.phase_name]
        param_search = dbe.search
        theta_param_query = (
            (where('phase_name') == phase.name) & \
            (where('parameter_type') == 'THETA') & \
            (where('constituent_array').test(self._array_validity))
        )
        lntheta = self.redlich_kister_sum(phase, param_search, theta_param_query)
        theta = exp(lntheta)
        if lntheta != 0:
            result = 1.5*v.R*theta + 3*v.R*v.T*log(1-exp(-theta/v.T))
        else:
            result = 0
        return result / self._site_ratio_normalization

    @staticmethod
    def _quasi_mole_fraction(species_name, phase_name, constituent_array,
                             site_ratios,
                             substitutional_sublattice_idxs,
                             ):
        """
        Return an abstract syntax tree of the quasi mole fraction of the
        given species as a function of this phases's constituent site fractions.

        These mole fractions are "quasi" mole fractions because

        1. Vacancies are treated as regular species - they have mole fractions
           defined and the site fraction of vacancies are not used to normalize
           the mole fractions of the real constituents by the 1 - y_{VA} factor.
        2. The mole fractions are only computed over the sublattices that
           participate in the ordering/disordering. Species in non-ordering
           ("interstitial") sublattices do not contribute to the mole fractions
           that replace the site fractions.

        These constraints ensures that the ordering energy goes to zero when the
        substitutional sublattice is disordered, regardless of the occupancy of
        the interstitial sublattice.
        """

        # Normalize site ratios
        site_ratio_normalization = 0
        numerator = S.Zero
        for idx, sublattice in enumerate(constituent_array):
            # only count species from substitutional sublattices
            if idx not in substitutional_sublattice_idxs:
                continue
            if species_name in list(sublattice):
                site_ratio_normalization += site_ratios[idx]
                numerator += site_ratios[idx] * \
                    v.SiteFraction(phase_name, idx, species_name)

        if site_ratio_normalization == 0 and species_name.name == 'VA':
            return 1

        if site_ratio_normalization == 0:
            raise ValueError(
                f'Couldn\'t find {species_name} in a substitutional sublattice '
                f'(indices: {substitutional_sublattice_idxs}) '
                f'of the constituents {constituent_array}'
                )

        return numerator / site_ratio_normalization

    @staticmethod
    def _partitioned_expr(disord_expr, ord_expr, disordered_mole_fraction_dict, ordered_mole_fraction_dict):
        """Return the expression from adding the disordered part and ordering part

        Given expressions E^{dis}(y^{dis}_i) and E^{ord}(y^{ord}_i), return:

            E^{dis}(x^{ord}_i) + (E^{ord}(y^{ord}_i) - E^{ord}(y^{ord}_i = x^{ord}_i))

        where:

        * y^{dis}_i are the site fractions of the disordered phase
        * y^{ord}_i are the site fractions of the ordered phase
        * x^{ord}_i are the quasi mole fractions of the ordered phase (in terms
             of the ordered phase site fractions)

        """
        disord_expr = disord_expr.xreplace(disordered_mole_fraction_dict)
        ordering_expr = ord_expr - ord_expr.xreplace(ordered_mole_fraction_dict)
        return disord_expr + ordering_expr

    def atomic_ordering_energy(self, dbe):
        """
        Return the atomic ordering contribution in symbolic form.

        If the current phase is anything other than the ordered phase in a
        paritioned order/disorder Gibbs energy model, this method will return
        zero. If the current phase is the ordered phase, ordering energy is
        computed by equation (18) of Connetable *et al.* [1]_:
        :math:`\Delta G^\mathrm{ord}(y_i) = G^\mathrm{ord}(y_i) - G^\mathrm{ord}(y_i = x_i)`

        This method must be the last energy contribution called because it plays
        several roles that require all other contributions to be defined:

           1. The current AST in self.models represents the ordered energy
           :math:`G^\mathrm{ord}(y_i)`. To compute the ordering energy, all
           contributions to the ordered energy must have already been counted.

           2. The true energy of the phase should be the sum of the disordered
           phase's energy and the ordering energy. That is,
           :math:`G = G^\mathrm{dis} + \Delta G^\mathrm{ord}(y_i)`. This method
           not only computes the ordering energy, but also replaces the other
           model contributions by the disordered phase's energy.

           3. Physical properties are partitioned in the same way as the
           energy. See Section 5.8.6 of Lukas, Fries and Sundman [2]_.

        Notes
        -----
        .. caution::
           This method overwrites the ``self.models`` dictionary with the model
           contributions for the disordered phase.

        This method assumes that the first sublattice of the disordered phase is
        the substitutional sublattice and all other sublattices are
        interstitial. In the ordered phase, all sublattices with constituents
        that match the disordered substitutional sublattice will be treated as
        disordered (with site fractions replaced by quasi mole fractions in the
        ordered sublattices) and the interstitial sublattices will not have any
        site fractions substituted.

        References
        ----------

        .. [1] Connetable et al., Calphad 2008, 32 (2), 361–370. doi: 10.1016/j.calphad.2008.01.002
        .. [2] Lukas, Fries, and Sundman, Computational Thermodynamics: the Calphad Method, Cambridge University Press (2007).

        """
        phase = dbe.phases[self.phase_name]
        ordered_phase_name = phase.model_hints.get('ordered_phase', None)
        disordered_phase_name = phase.model_hints.get('disordered_phase', None)
        if phase.name != ordered_phase_name:
            return S.Zero
        ordered_phase = dbe.phases[ordered_phase_name]
        constituents = [sorted(set(c).intersection(self.components)) for c in ordered_phase.constituents]
        disordered_phase = dbe.phases[disordered_phase_name]
        disordered_model = self.__class__(dbe, sorted(self.components), disordered_phase_name)

        # Get substitutional sublattice indices (for the ordered phase) and
        # validate that the number of interstitial sublattices is consistent
        # with the disordered phase.
        # Assumes first sublattice of the disordered phase is the sublattice
        # that can be come ordered:
        disordered_subl_constituents = disordered_phase.constituents[0]
        ordered_constituents = ordered_phase.constituents
        substitutional_sublattice_idxs = []
        for idx, subl_constituents in enumerate(ordered_constituents):
            # Assumes that the ordered phase sublattice describes the ordering
            # if it has exactly the same constituents. Could be a source of
            # false positives if any interstitial sublattices have the same
            # constituents as the disordered sublattice, but there's not an
            # explicit way to specify which sublattices are ordering. We try to
            # compensate for this assumption by validating (next).
            if len(disordered_subl_constituents.symmetric_difference(subl_constituents)) == 0:
                substitutional_sublattice_idxs.append(idx)
        # validate
        num_substitutional_sublattice_idxs = len(substitutional_sublattice_idxs)
        num_ordered_interstitial_subls = len(ordered_phase.sublattices) - num_substitutional_sublattice_idxs
        num_disordered_interstitial_subls = len(disordered_phase.sublattices) - 1
        if num_ordered_interstitial_subls != num_disordered_interstitial_subls:
            raise ValueError(
                f'Number of interstitial sublattices for the disordered phase '
                f'({num_disordered_interstitial_subls}) and the ordered phase '
                f'({num_ordered_interstitial_subls}) do not match. Got '
                f'substitutional sublattice indices of {substitutional_sublattice_idxs}.'
                )
        # We also validate that no physical properties have ordered
        # contributions because the underlying physical property needs to
        # paritioned and substituted for the physical property in the disordered
        # expression. This can be safely removed when partitioned
        # physical properties are correctly substituted into the disordered
        # energy.
        for contrib, value in self.models.items():
            # To handle ordering in user-defined subclasses, we assume that all properties
            # that are not reference, ideal, or excess are physical contributions.
            if contrib in ('ref', 'idmix', 'xsmix'):
                continue
            if value != S.Zero:
                warnings.warn(
                    f"The order-disorder model for \"{self.phase_name}\" has a contribution from "
                    f"the physical property model `{dict(self.contributions)[contrib]}`. "
                    f"Partitioned physical properties are not correctly substituted into the "
                    f"disordered part of the energy. THE GIBBS ENERGY CALCULATED FOR THIS PHASE "
                    f"MAY BE INCORRECT. Please see the discussion in "
                    f"https://github.com/pycalphad/pycalphad/pull/311 for more details."
                    )

        # Save all of the ordered energy contributions
        # Needs to extract a copy of self.models.values because the values will
        # be updated to the disordered energy contributions later
        ordered_energy = Add(*list(self.models.values()))

        # Compute the molefraction_dict, which will map ordered phase site
        # fractions to the quasi mole fractions representing the disordered state
        molefraction_dict = {}
        ordered_sitefracs = [x for x in ordered_energy.free_symbols if isinstance(x, v.SiteFraction)]
        for sitefrac in ordered_sitefracs:
            if sitefrac.sublattice_index in substitutional_sublattice_idxs:
                molefraction_dict[sitefrac] = \
                    self._quasi_mole_fraction(sitefrac.species,
                                              ordered_phase_name,
                                              constituents,
                                              ordered_phase.sublattices,
                                              substitutional_sublattice_idxs,
                                              )

        # Compute the variable_rename_dict, which will map disordered phase site
        # fractions to the quasi mole fractions representing the disordered state
        variable_rename_dict = {}
        disordered_sitefracs = [x for x in disordered_model.energy.free_symbols if isinstance(x, v.SiteFraction)]
        for atom in disordered_sitefracs:
            if atom.sublattice_index == 0:  # only the first sublattice is substitutional
                variable_rename_dict[atom] = \
                    self._quasi_mole_fraction(atom.species,
                                              ordered_phase_name,
                                              constituents,
                                              ordered_phase.sublattices,
                                              substitutional_sublattice_idxs,
                                              )

            else:
                shifted_subl_index = atom.sublattice_index + num_substitutional_sublattice_idxs - 1
                variable_rename_dict[atom] = \
                    v.SiteFraction(ordered_phase_name, shifted_subl_index, atom.species)

        # 1: Compute the ordering energy
        # Step 2 will put the disordered parts into the correct model
        # contributions. There's no technical reason for doing it this way
        # compared to setting the AST to the _partitioned_expr for the total
        # energy - this is more for bookkeeping of the model contributions.
        ordering_energy = self._partitioned_expr(S.Zero, ordered_energy, {}, molefraction_dict)

        # 2: Replace the ordered energy contributions with the disordered contributions
        self.models.clear()
        for name, value in disordered_model.models.items():
            self.models[name] = value.xreplace(variable_rename_dict)

        # 3: Handle physical properties, these also are contributed to by the
        # disordered phase *and* an "ordering" contribution. For now, we only
        # handle the magnetic parameters, since the other parameters are not
        # stored as properties (e.g. Einstein THETA).
        # TODO: Note that these do not affect the Gibbs energy expression!
        # The disordered model's energetic contribution from physical
        # properties needs to use the partitioned property in the disordered
        # energy contribution. This is not possible at the time of writing.
        self.TC = self.curie_temperature = self._partitioned_expr(disordered_model.TC, self.TC, variable_rename_dict, molefraction_dict)
        self.BMAG = self.beta = self._partitioned_expr(disordered_model.BMAG, self.BMAG, variable_rename_dict, molefraction_dict)
        self.NT = self.neel_temperature = self._partitioned_expr(disordered_model.NT, self.NT, variable_rename_dict, molefraction_dict)

        return ordering_energy

    # TODO: fix case for VA interactions: L(PHASE,A,VA:VA;0)-type parameters
    def shift_reference_state(self, reference_states, dbe, contrib_mods=None, output=('GM', 'HM', 'SM', 'CPM'), fmt_str="{}R"):
        """
        Add new attributes for calculating properties w.r.t. an arbitrary pure element reference state.

        Parameters
        ----------
        reference_states : Iterable of ReferenceState
            Pure element ReferenceState objects. Must include all the pure
            elements defined in the current model.
        dbe : Database
            Database containing the relevant parameters.
        output : Iterable, optional
            Parameters to subtract the ReferenceState from, defaults to ('GM', 'HM', 'SM', 'CPM').
        contrib_mods : Mapping, optional
            Map of {model contribution: new value}. Used to adjust the pure
            reference model contributions at the time this is called, since
            the `models` attribute of the pure element references are
            effectively static after calling this method.
        fmt_str : str, optional
            String that will be formatted with the `output` parameter name.
            Defaults to "{}R", e.g. the transformation of 'GM' -> 'GMR'

        """
        # Error checking
        # We ignore the case that the ref states are overspecified (same ref states can be used in different models w/ different active pure elements)
        model_pure_elements = set(get_pure_elements(dbe, self.components))
        refstate_pure_elements_list = get_pure_elements(dbe, [r.species for r in reference_states])
        refstate_pure_elements = set(refstate_pure_elements_list)
        if len(refstate_pure_elements_list) != len(refstate_pure_elements):
            raise DofError("Multiple ReferenceState objects exist for at least one pure element: {}".format(refstate_pure_elements_list))
        if not refstate_pure_elements.issuperset(model_pure_elements):
            raise DofError("Non-existent ReferenceState for pure components {} in {} for {}".format(model_pure_elements.difference(refstate_pure_elements), self, self.phase_name))

        contrib_mods = contrib_mods or {}

        def _pure_element_test(constituent_array):
            all_comps = set()
            for sublattice in constituent_array:
                if len(sublattice) != 1:
                    return False
                all_comps.add(sublattice[0].name)
            pure_els = all_comps.intersection(model_pure_elements)
            return len(pure_els) == 1

        # Remove interactions from a copy of the Database, avoids any element/VA interactions.
        endmember_only_dbe = copy.deepcopy(dbe)
        endmember_only_dbe._parameters.remove(~where('constituent_array').test(_pure_element_test))
        reference_dict = {out: [] for out in output}  # output: terms list
        for ref_state in reference_states:
            if ref_state.species not in self.components:
                continue
            mod_pure = self.__class__(endmember_only_dbe, [ref_state.species, v.Species('VA')], ref_state.phase_name, parameters=self._parameters_arg)
            # apply the modifications to the Models
            for contrib, new_val in contrib_mods.items():
                mod_pure.models[contrib] = new_val
            # set all the free site fractions to one, this should effectively delete any mixing terms spuriously added, e.g. idmix
            site_frac_subs = {sf: 1 for sf in mod_pure.ast.free_symbols if isinstance(sf, v.SiteFraction)}
            for mod_key, mod_val in mod_pure.models.items():
                mod_pure.models[mod_key] = self.symbol_replace(mod_val, site_frac_subs)
            moles = self.moles(ref_state.species)
            # get the output property of interest, substitute the fixed state variables (e.g. T=298.15) and add the pure element moles weighted term to the list of terms
            # substitution of fixed state variables has to happen after getting the attribute in case there are any derivatives involving that state variable
            for out in reference_dict.keys():
                mod_out = self.symbol_replace(getattr(mod_pure, out), ref_state.fixed_statevars)
                reference_dict[out].append(mod_out*moles)

        # set the attribute on the class
        for out, terms in reference_dict.items():
            reference_contrib = Add(*terms)
            referenced_value = getattr(self, out) - reference_contrib
            setattr(self, fmt_str.format(out), referenced_value)


class TestModel(Model):
    """
    Test Model object for global minimization.

    Equation 15.2 in:
    P.M. Pardalos, H.E. Romeijn (Eds.), Handbook of Global Optimization,
    vol. 2. Kluwer Academic Publishers, Boston/Dordrecht/London (2002)

    Parameters
    ----------
    dbf : Database
        Ignored by TestModel but retained for API compatibility.
    comps : sequence
        Names of components to consider in the calculation.
    phase : str
        Name of phase model to build.
    solution : sequence, optional
        Float array locating the true minimum. Same length as 'comps'.
        If not specified, randomly generated and saved to self.solution

    Methods
    -------
    None yet.

    Examples
    --------
    None yet.
    """
    def __init__(self, dbf, comps, phase, solution=None, kmax=None):
        self.components = set(comps)
        if 'VA' in self.components:
            raise ValueError('Vacancies are unsupported in TestModel')
        self.models = dict()
        variables = [v.SiteFraction(phase.upper(), 0, x) for x in sorted(self.components)]
        if solution is None:
            solution = np.random.dirichlet(np.ones_like(variables, dtype=np.int_))
        self.solution = dict(list(zip(variables, solution)))
        kmax = kmax if kmax is not None else 2
        scale_factor = 1e4 * len(self.components)
        ampl_scale = 1e3 * np.ones(kmax, dtype=np.float_)
        freq_scale = 10 * np.ones(kmax, dtype=np.float_)
        polys = Add(*[ampl_scale[i] * sin(freq_scale[i] * Add(*[Add(*[(varname - sol)**(j+1)
                                                                      for varname, sol in self.solution.items()])
                                                                for j in range(kmax)]))**2
                      for i in range(kmax)])
        self.models['test'] = scale_factor * Add(*[(varname - sol)**2 for varname, sol in self.solution.items()]) + polys
