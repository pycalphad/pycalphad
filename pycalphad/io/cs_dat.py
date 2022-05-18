"""
Support for reading ChemSage DAT files.
"""

from typing import Dict, List, Optional, Tuple
import re
import warnings
import numpy as np
import itertools
from dataclasses import dataclass
from collections import deque
from symengine import S, log, Piecewise, And
from pycalphad import Database, variables as v
from .grammar import parse_chemical_formula

# From ChemApp Documentation, section 11.1 "The format of a ChemApp data-file"
# We use a leading zero term because the data file's indices are 1-indexed and
# this prevents us from needing to shift the indicies.
# Exponents are in floating point so that round-trip write/read passes equality checks
GIBBS_TERMS = (S.Zero, S.One, v.T, v.T*log(v.T), v.T**2.0, v.T**3.0, v.T**(-1.0))
CP_TERMS = (S.Zero, S.One, v.T, v.T**2.0, v.T**(-2.0))
EXCESS_TERMS = (S.Zero, S.One, v.T, v.T*log(v.T), v.T**2.0, v.T**3.0, v.T**(-1.0), v.P, v.P**2.0)
DEFAULT_T_MIN = 0.01  # The same as for TDBs when no minimum temperature is given.

def _parse_species_postfix_charge(formula) -> v.Species:
    name = formula
    # handle postfix charge: FE[2+] CU[+] CL[-] O[2-]
    match = re.search(r'\[([0-9]+)?([-+])\]', formula)
    if match is not None:
        # remove the charge from the formula
        formula = formula[:match.start()]
        charge = int(f'{match.groups()[1]}{match.groups()[0] or 1}')
    else:
        charge = 0
    # assumes that the remaining formula is a pure element
    constituents = dict(parse_chemical_formula(formula)[0])
    return v.Species(name, constituents=constituents, charge=charge)

class TokenParserError(Exception):
    """Exception raised when the TokenParser hits a parsing error."""
    pass

class TokenParser():
    def __init__(self, string):
        self._line_number = 1  # user facing, so make it one-indexed
        self._lines_deque = deque(string.split("\n"))
        self._current_line = self._lines_deque.popleft()
        self._tokens_deque = deque(self._current_line.split())
    
    def __getitem__(self, i: int):
        # Instantiate a new TokenParser for the current state so we can look ahead without messing up our line numbers
        lines = "\n".join(deque([" ".join(self._tokens_deque)]) + self._lines_deque)
        tmp_parser = TokenParser(lines)
        if i > 0:
            for _ in range(i - 1):
                token = tmp_parser._next()
        return tmp_parser._next()

    def _next(self):
        try:
            token = self._tokens_deque.popleft()
        except IndexError as e:
            # If we're out of tokens, get the next line and try to grab a token again
            self._current_line = self._lines_deque.popleft()
            self._line_number += 1
            self._tokens_deque.extend(self._current_line.split())
            # call next instead of popleft() on the deque in case self._current_line has no tokens
            token = self._next()
        return token

    def parse(self, cls: type):
        next_token = self._next()
        try:
            obj = cls(next_token)
        except ValueError as e:
            # Return the token and re-raise with a ParseError 
            self._tokens_deque.appendleft(next_token)
            raise TokenParserError(f"Error at line number {self._line_number + 1}: {e.args} for line:\n    {self._current_line}") from e
        else:
            return obj

    def parseN(self, N: int, cls: type):
        if N < 1:
            raise ValueError(f'N must be >=1, got {N}')
        return [self.parse(cls) for _ in range(N)]


@dataclass
class Header:
    list_soln_species_count: List[int]
    num_stoich_phases: int
    pure_elements: List[str]
    pure_elements_mass: List[float]
    gibbs_coefficient_idxs: List[int]
    excess_coefficient_idxs: List[int]


@dataclass
class AdditionalCoefficientPair:
    coefficient: float
    exponent: float

    def expr(self):
        if self.exponent == 99:
            # this is a special case and means log(v.T)
            # See ChemApp documentation, section 11.1 cosi.dat, Line 5
            return self.coefficient*log(v.T)
        if abs(self.exponent) > 9:
            warnings.warn(f"Additional coefficient pair has an exponent of {self.exponent}, which should be between -9 and +9.")
        return self.coefficient * v.T**(self.exponent)


@dataclass
class PTVmTerms:
    terms: List[float]


@dataclass
class IntervalBase:
    T_max: float

    def expr(self):
        raise NotImplementedError("Subclasses of IntervalBase must define an expression for the energy")

    def cond(self, T_min=DEFAULT_T_MIN):
        if T_min == self.T_max:
            # To avoid an impossible, always False condition an open interval
            # is assumed. We choose 10000 K as the dummy (as in TDBs).
            return And((T_min <= v.T), (v.T < 10000))
        return And((T_min <= v.T), (v.T < self.T_max))

    def expr_cond_pair(self, *args, T_min=DEFAULT_T_MIN, **kwargs):
        """Return an (expr, cond) tuple used to construct Piecewise expressions"""
        expr = self.expr(*args, **kwargs)
        cond = self.cond(T_min)
        return (expr, cond)


@dataclass
class IntervalG(IntervalBase):
    coefficients: List[float]
    additional_coeff_pairs: List[AdditionalCoefficientPair]
    PTVm_terms: List[PTVmTerms]

    def expr(self, indices):
        """Return an expression for the energy in this temperature interval"""
        energy = S.Zero
        # Add fixed energy terms
        energy += sum([C*GIBBS_TERMS[i] for C, i in zip(self.coefficients, indices)])
        # Add additional energy coefficient-exponent pair terms
        energy += sum([addit_term.expr() for addit_term in self.additional_coeff_pairs])
        # P-T molar volume terms, not supported
        if len(self.PTVm_terms) > 0:
            raise NotImplementedError("P-T molar volume terms are not supported")
        return energy


# TODO: not yet supported
@dataclass
class IntervalCP(IntervalBase):
    # Fixed term heat capacity interval with extended terms
    H298: float
    S298: float
    CP_coefficients: float
    H_trans: float
    additional_coeff_pairs: List[AdditionalCoefficientPair]
    PTVm_terms: List[PTVmTerms]

    def expr(self, indices, T_min=DEFAULT_T_MIN):
        """Return an expression for the energy in this temperature interval"""
        raise NotImplementedError("Heat capacity descriptions of the Gibbs energy are not implemented.")


@dataclass
class Endmember():
    species_name: str
    gibbs_eq_type: str
    stoichiometry_pure_elements: List[float]
    intervals: List[IntervalBase]

    def expr(self, indices):
        """Return a Piecewise (in temperature) energy expression for this endmember (i.e. only the data from the energy intervals)"""
        T_min = DEFAULT_T_MIN
        expr_cond_pairs = []
        for interval in self.intervals:
            expr_cond_pairs.append(interval.expr_cond_pair(indices, T_min=T_min))
            T_min = interval.T_max
        # a (expr, True) condition must be at the end
        expr_cond_pairs.append((S.Zero, S.true))
        return Piecewise(*expr_cond_pairs)

    def constituents(self, pure_elements: List[str]) -> Dict[str, float]:
        return {el: amnt for el, amnt in zip(pure_elements, self.stoichiometry_pure_elements) if amnt != 0.0}

    def constituent_array(self) -> List[List[str]]:
        return [[sp] for sp in self.species_name.split(':')]

    def species(self, pure_elements) -> List[v.Species]:
        """Return an unordered list of Species objects detected in this endmember"""
        if len(self.species_name.split(':')) > 1:
            # If given in sublattice notation, assume species are pure elements
            # i.e. multi-sublattice models cannot have associates
            all_species = self.species_name.split(':')
            return np.unique([v.Species(sp_str) for sp_str in all_species]).tolist()
        else:
            # We only have one sublattice, this can be a non-pure element species
            return [v.Species(self.species_name, constituents=self.constituents(pure_elements))]

    def insert(self, dbf: Database, phase_name: str, constituent_array: List[List[str]], gibbs_coefficient_idxs: List[int]):
        dbf.add_parameter('G', phase_name, constituent_array, 0, self.expr(gibbs_coefficient_idxs), force_insert=False)

@dataclass
class EndmemberQKTO(Endmember):
    stoichiometric_factor: float
    chemical_group: int

    def insert(self, dbf: Database, phase_name: str, constituent_array: List[List[str]], gibbs_coefficient_idxs: List[int]):
        dbf.add_parameter('G', phase_name, constituent_array, 0, self.expr(gibbs_coefficient_idxs), force_insert=False)
        # Most databases in the wild use stoichiometric factors of unity,
        # so we're avoiding the complexity of non-unity factors for now.
        if not np.isclose(self.stoichiometric_factor, 1.0):
            raise ValueError(f"QKTO endmembers with stoichiometric factors other than 1 are not yet supported. For {phase_name}: got {self.stoichiometric_factor} for {self}")


@dataclass
class EndmemberMagnetic(Endmember):
    curie_temperature: float
    magnetic_moment: float

    def insert(self, dbf: Database, phase_name: str, pure_elements: List[str], gibbs_coefficient_idxs: List[int]):
        # add Gibbs energy
        super().insert(dbf, phase_name, pure_elements, gibbs_coefficient_idxs)

        # also add magnetic parameters
        dbf.add_parameter('BMAGN', phase_name, self.constituent_array(),
                          0, self.magnetic_moment, force_insert=False)
        dbf.add_parameter('TC', phase_name, self.constituent_array(),
                          0, self.curie_temperature, force_insert=False)


@dataclass
class EndmemberRealGas(Endmember):
    # Tsonopoulos data
    Tc: float
    Pc: float
    Vc: float
    acentric_factor: float
    dipole_moment: float

    def insert(*args, **kwargs):
        raise NotImplementedError("Inserting parameters for real gas Endmembers is not supported.")


@dataclass
class EndmemberAqueous(Endmember):
    charge: float

    def insert(*args, **kwargs):
        raise NotImplementedError("Inserting parameters for aqueous Endmembers is not supported.")


@dataclass
class ExcessBase:
    interacting_species_idxs: List[int]

    def _map_const_idxs_to_subl_idxs(self, num_subl_species: List[int]) -> List[List[int]]:
        """
        Converts from one-indexed linear phase species indices to zero-indexed
        sublattice species indices.

        Parameters
        ----------
        num_subl_species: List[int]
            Number of species in each sublattice, i.e. [1, 2, 1] could
            correspond to a sublattice model of [['A'], ['A', 'B'], ['C']]

        Returns
        -------
        List[List[int]] - a list of species index lists

        """
        cum_num_subl_species = np.cumsum(num_subl_species)
        # initialize an empty sublattice model
        subl_species_idxs = [[] for _ in range(len(num_subl_species))]
        for linear_idx in self.interacting_species_idxs:
            # Find which sublattice this species belongs to by seeing that the
            # linear_idx is contained within the cumulative number of species
            # on each sublattice
            subl_idx = (cum_num_subl_species >= linear_idx).tolist().index(True)
            # Determine the index of the species within the sublattice
            # This is still the one-indexed value
            if subl_idx == 0:
                subl_sp_idx = linear_idx
            else:
                subl_sp_idx = linear_idx - cum_num_subl_species[subl_idx - 1]
            # convert one-indexed species index to zero-indexed
            subl_sp_idx -= 1
            # add the species index to the right sublattice
            subl_species_idxs[subl_idx].append(subl_sp_idx)
        # all sublattices must be occupied
        assert all(len(subl) > 0 for subl in subl_species_idxs)
        return subl_species_idxs

    def constituent_array(self, phase_constituents: List[List[str]]) -> List[List[str]]:
        """
        Return the constituent array of this interaction using the entire phase
        sublattice model.

        This doesn't take into account any re-ordering of the phase_constituents
        or interacting_species_idxs. All mapping on to proper v.Species objects
        occurs in Database.add_parameter.

        Examples
        --------
        >>> phase_constituents = [['A'], ['A', 'B'], ['A', 'B', 'C']]
        >>> ex = ExcessBase([1, 2, 4, 6])
        >>> ix_const_arr = ex.constituent_array(phase_constituents)
        >>> assert ix_const_arr == [['A'], ['A'], ['A', 'C']]
        """
        num_subl_species = [len(subl) for subl in phase_constituents]
        subl_species_idxs = self._map_const_idxs_to_subl_idxs(num_subl_species)
        return [[phase_constituents[subl_idx][idx] for idx in subl] for subl_idx, subl in enumerate(subl_species_idxs)]

    def insert(self, dbf: Database, phase_name: str, phase_constituents: List[List[str]], excess_coefficient_idxs: List[int]):
        raise NotImplementedError(f"Subclass {type(self).__name__} of ExcessBase must implement `insert` to add the phase, constituents and parameters to the Database.")


@dataclass
class ExcessRKM(ExcessBase):
    """
    Examples
    --------
    >>> assert ExcessRKM([1, 2, 3, 4], 0, [0])._map_const_idxs_to_subl_idxs([2, 3]) == [[0, 1], [0, 1]]
    >>> assert ExcessRKM([1, 3, 4], 0, [0])._map_const_idxs_to_subl_idxs([2, 3]) == [[0], [0, 1]]
    >>> assert ExcessRKM([1, 2], 0, [0])._map_const_idxs_to_subl_idxs([4]) == [[0, 1]]
    >>> assert ExcessRKM([1, 2, 3], 0, [0])._map_const_idxs_to_subl_idxs([3]) == [[0, 1, 2]]
    >>> assert ExcessRKM([1, 2, 3, 4], 0, [0])._map_const_idxs_to_subl_idxs([1, 1, 2]) == [[0], [0], [0, 1]]
    """
    parameter_order: int
    coefficients: List[float]

    def expr(self, indices):
        """Return an expression for the energy in this temperature interval"""
        energy = S.Zero
        # Add fixed energy terms
        energy += sum([C*EXCESS_TERMS[i] for C, i in zip(self.coefficients, indices)])
        return energy

    def insert(self, dbf: Database, phase_name: str, phase_constituents: List[List[str]], excess_coefficient_idxs: List[int]):
        """
        Requires all Species in dbf.species to be defined.
        """
        # Note: Thermochimica does _not_ sort species alphabetically (as is done by TDB formats),
        # so a constituent array of ("A", "B") != ("B", "A") for odd order terms.
        const_array = self.constituent_array(phase_constituents)
        dbf.add_parameter('L', phase_name, const_array, self.parameter_order, self.expr(excess_coefficient_idxs), force_insert=False)


@dataclass
class ExcessRKMMagnetic(ExcessBase):
    parameter_order: int
    curie_temperature: float
    magnetic_moment: float

    def insert(self, dbf: Database, phase_name: str, phase_constituents: List[List[str]], excess_coefficient_idxs: List[int]):
        """
        Requires all Species in dbf.species to be defined.
        """
        # See the comment about sorting in ExcessRKM
        const_array = self.constituent_array(phase_constituents)
        dbf.add_parameter('TC', phase_name, const_array, self.parameter_order, self.curie_temperature, force_insert=False)
        dbf.add_parameter('BMAGN', phase_name, const_array, self.parameter_order, self.magnetic_moment, force_insert=False)


@dataclass
class ExcessQKTO(ExcessBase):
    exponents: List[int]
    coefficients: List[float]

    def expr(self, indices):
        """Return an expression for the energy in this temperature interval"""
        energy = S.Zero
        # Add fixed energy terms
        energy += sum([C*EXCESS_TERMS[i] for C, i in zip(self.coefficients, indices)])
        return energy

    def insert(self, dbf: Database, phase_name: str, phase_constituents: List[str], excess_coefficient_idxs: List[int]):
        const_array = self.constituent_array(phase_constituents)
        exponents = [exponent - 1 for exponent in self.exponents]  # For some reason, an exponent of 1 really means an exponent of zero
        dbf.add_parameter(
            "QKT", phase_name, const_array, param_order=None,
            param=self.expr(excess_coefficient_idxs), exponents=exponents,
            force_insert=False,
            )

@dataclass
class PhaseBase:
    phase_name: str
    phase_type: str
    endmembers: List[Endmember]

    def insert(self, dbf: Database, pure_elements: List[str], gibbs_coefficient_idxs: List[int], excess_coefficient_idxs: List[int]):
        """Enter this phase and its parameters into the Database.

        This method should call:

        * `dbf.add_phase`
        * `dbf.structure_entry`
        * `dbf.add_phase_constituents`
        * `dbf.add_parameter` for all parameters

        """
        raise NotImplementedError(f"Subclass {type(self).__name__} of PhaseBase must implement `insert` to add the phase, constituents and parameters to the Database.")


@dataclass
class Phase_Stoichiometric(PhaseBase):
    magnetic_afm_factor: Optional[float]
    magnetic_structure_factor: Optional[float]

    def insert(self, dbf: Database, pure_elements: List[str], gibbs_coefficient_idxs: List[int], excess_coefficient_idxs: List[int]):
        model_hints = {}
        if self.magnetic_afm_factor is not None or self.magnetic_structure_factor is not None:
            # This follows the Redlich-Kister Muggianu IHJ model. The ChemSage
            # docs don't indicate that it's an IHJ model, but Eriksson and Hack,
            # Met. Trans. B 21B (1990) 1013 says that it follows IHJ.
            model_hints['ihj_magnetic_structure_factor'] = self.magnetic_structure_factor
            # The TDB syntax would define the AFM factor for FCC as -3
            # while ChemSage defines +0.333. This is likely because the
            # model divides by the AFM factor (-1/3). We convert the AFM
            # factor to the version used in the TDB/Model.
            model_hints['ihj_magnetic_afm_factor'] = -1/self.magnetic_afm_factor

        assert len(self.endmembers) == 1  # stoichiometric phase

        # For stoichiometric endmembers, the endmember "constituent array" is
        # just the phase name. We can just define the real constituent array in
        # terms of pure elements, where each element gets it's own sublattice.
        constituent_dict = self.endmembers[0].constituents(pure_elements)
        constituent_array = [[el] for el in sorted(constituent_dict.keys())]
        subl_stoich_ratios = [constituent_dict[el] for el in sorted(constituent_dict.keys())]

        dbf.add_phase(self.phase_name, model_hints=model_hints, sublattices=subl_stoich_ratios)
        dbf.add_structure_entry(self.phase_name, self.phase_name)
        dbf.add_phase_constituents(self.phase_name, constituent_array)
        self.endmembers[0].insert(dbf, self.phase_name, constituent_array, gibbs_coefficient_idxs)


@dataclass
class Phase_CEF(PhaseBase):
    subl_ratios: List[float]
    constituent_array: List[List[str]]
    endmember_constituent_idxs: List[List[int]]
    excess_parameters: List[ExcessBase]
    magnetic_afm_factor: Optional[float]
    magnetic_structure_factor: Optional[float]

    def insert(self, dbf: Database, pure_elements: List[str], gibbs_coefficient_idxs: List[int], excess_coefficient_idxs: List[int]):
        model_hints = {}
        if self.magnetic_afm_factor is not None and self.magnetic_structure_factor is not None:
            # This follows the Redlich-Kister Muggianu IHJ model. The ChemSage
            # docs don't indicate that it's an IHJ model, but Eriksson and Hack,
            # Met. Trans. B 21B (1990) 1013 says that it follows IHJ.
            model_hints['ihj_magnetic_structure_factor'] = self.magnetic_structure_factor
            # The TDB syntax would define the AFM factor for FCC as -3
            # while ChemSage defines +0.333. This is likely because the
            # model divides by the AFM factor (-1/3). We convert the AFM
            # factor to the version used in the TDB/Model.
            model_hints['ihj_magnetic_afm_factor'] = -1/self.magnetic_afm_factor
        if any(isinstance(xs, (ExcessQKTO)) for xs in self.excess_parameters) and any(isinstance(xs, (ExcessRKM)) for xs in self.excess_parameters):
            raise ValueError("ExcessQKTO and ExcessRKM parameters found, but they cannot co-exist.")

        # Try adding model hints for chemical groups from endmembers
        chemical_groups = {}
        for endmember in self.endmembers:
            if hasattr(endmember, "chemical_group"):
                endmember_species = endmember.species(pure_elements)
                # make the assumption that there's only one species in this endmember
                # currently, only QKTO model endmembers supply chemical groups
                # and QKTO models in the DAT can only have one sublattice.
                species = endmember_species[0]
                if species in chemical_groups:
                    raise ValueError(f"Species {species} is already present in the chemical groups dictionary for phase {self.phase_name}  with endmembers {self.endmembers}.")
                else:
                    chemical_groups[species] = endmember.chemical_group
        if len(chemical_groups.keys()) > 0:
            model_hints["chemical_groups"] = chemical_groups

        dbf.add_phase(self.phase_name, model_hints=model_hints, sublattices=self.subl_ratios)
        dbf.add_structure_entry(self.phase_name, self.phase_name)

        # This does two things:
        # 1. set the self.constituent_array
        # 2. add species to the database
        if self.constituent_array is None:
            # Before we add parameters, we need to first add all the species to dbf,
            # since dbf.add_parameter takes a constituent array of string species
            # names which are mapped to Species objects
            for endmember in self.endmembers:
                for sp in endmember.species(pure_elements):
                    invalid_shared_names = [(sp.name == esp.name and sp != esp) for esp in dbf.species]
                    if any(invalid_shared_names):
                        # names match some already  but constituents do not
                        raise ValueError(f"A Species named {sp.name} (defined for phase {self.phase_name}) already exists in the database's species ({dbf.species}), but the constituents do not match.")
                    dbf.species.add(sp)

            # Construct constituents for this phase, this loop could be merged with
            # the parameter additions above (it's not dependent like the species
            # step), but we are keeping it logically separate to make it clear how
            # it's working. This assumes that all constituents are present in
            # endmembers (i.e. there are no endmembers that are implicit).

            constituents = [[] for _ in range(len(self.subl_ratios))]
            for endmember in self.endmembers:
                for subl, const_subl in zip(endmember.constituent_array(), constituents):
                    const_subl.extend(subl)
            self.constituent_array = constituents  # Be careful to preserve ordering here, since the mapping from species indices to species depends on the order of this
        else:
            # add the species to the database
            for subl in self.constituent_array:
                for const in subl:
                    dbf.species.add(_parse_species_postfix_charge(const))
        dbf.add_phase_constituents(self.phase_name, self.constituent_array)

        # Now that all the species are in the database, we are free to add the parameters
        if self.endmember_constituent_idxs is None:
            # we have to guess at the constituent array
            for endmember in self.endmembers:
                endmember.insert(dbf, self.phase_name, endmember.constituent_array(), gibbs_coefficient_idxs)
        else:
            # we know the constituent array from the indices and we don't have
            # to guess
            for endmember, const_idxs in zip(self.endmembers, self.endmember_constituent_idxs):
                em_const_array = [[self.constituent_array[i][sp_idx - 1]] for i, sp_idx in enumerate(const_idxs)]
                endmember.insert(dbf, self.phase_name, em_const_array, gibbs_coefficient_idxs)

        for excess_param in self.excess_parameters:
            excess_param.insert(dbf, self.phase_name, self.constituent_array, excess_coefficient_idxs)


def rename_element_charge(element, charge):
    """We use the _ to separate so we have something to split on."""
    if charge == 0:
        return f'{element}'
    elif charge > 0:
        return f'{element}+{charge}'
    else:
        return f'{element}-{abs(charge)}'


@dataclass
class SUBQPair(Endmember):
    stoichiometry_quadruplet: List[float]
    zeta: float

    def insert(self, dbf: Database, phase_name: str, constituent_array: List[str], gibbs_coefficient_idxs: List[int]):
        # Here the constituent array should be the pair name using the corrected
        # names, i.e. CU1.0CL1.0
        dbf.add_parameter(
            'MQMG', phase_name, constituent_array, param_order=None,
            param=self.expr(gibbs_coefficient_idxs), zeta=self.zeta,
            stoichiometry=self.stoichiometry_quadruplet,
            force_insert=False,
            )


@dataclass
class SUBQQuadrupletCoordinations:
    quadruplet_idxs: List[int]  # exactly four
    quadruplet_coordinations: List[float]  # exactly four

    def insert(self, dbf: Database, phase_name: str, As: List[str], Xs: List[str]):
        """Add a Z_i_AB:XY parameter for each species defined in the quadruplet"""
        linear_species = [''] + As + Xs  # the leading '' element pads for one-indexed quadruplet_idxs
        A, B, X, Y = tuple(linear_species[idx] for idx in self.quadruplet_idxs)
        Z_A, Z_B, Z_X, Z_Y = self.quadruplet_coordinations
        # Constituents and coordinations need to be canonically sorted (within each sublattice)
        constituent_array = []  # Should be split by sublattice, List[List[float]]
        coordinations = []  # Should be "linear", List[float]
        for const_subl, coord_subl in zip([[A, B], [X, Y]], [[Z_A, Z_B], [Z_X, Z_Y]]):
            sorted_const, sorted_coord = zip(*[(const, coord) for const, coord in sorted(zip(const_subl, coord_subl))])
            constituent_array.append(list(sorted_const))
            coordinations.extend(sorted_coord)
        dbf.add_parameter(
            "MQMZ", phase_name, constituent_array, param_order=None, param=None,
            coordinations=coordinations, force_insert=False,
            )


@dataclass
class SUBQExcessQuadruplet:
    mixing_type: int
    mixing_code: str  # G, Q, B, or R
    mixing_const: List[int]  # exactly four
    mixing_exponents: List[int]  # exactly four
    metadata: List[float]  # exactly twelve
    additional_cation_mixing_const: int
    additional_anion_mixing_const: int
    excess_coeffs: List[float]

    def expr(self, indices):
        """Return an expression for the energy in this temperature interval"""
        energy = S.Zero
        # Add fixed energy terms
        energy += sum([C*EXCESS_TERMS[i] for C, i in zip(self.excess_coeffs, indices)])
        return energy

    def insert(self, dbf: Database, phase_name: str, As: List[str], Xs: List[str], excess_coeff_indices: List[int]):
        linear_species = [None] + As + Xs  # the leading '' element pads for one-indexed quadruplet_idxs
        A, B, X, Y = tuple(linear_species[idx] for idx in self.mixing_const)
        constituent_array = [[A, B], [X, Y]]
        mixing_code = self.mixing_code
        exponents = self.mixing_exponents
        expr = self.expr(excess_coeff_indices)

        addtl_cation_mixing_const = linear_species[self.additional_cation_mixing_const]
        addtl_anion_mixing_const = linear_species[self.additional_anion_mixing_const]
        if addtl_cation_mixing_const is not None and addtl_anion_mixing_const is not None:
            raise ValueError(f"Having a cation _and_ anion as additional mixing constituents is not allowed. Got {addtl_cation_mixing_const} and {addtl_anion_mixing_const} for {phase_name} and quadruplet {A, B, X, Y}.")
        elif addtl_cation_mixing_const is not None:
            addtl_mixing_const = addtl_cation_mixing_const
            addtl_mixing_expon = exponents[2]
        elif addtl_anion_mixing_const is not None:
            addtl_mixing_const = addtl_anion_mixing_const
            addtl_mixing_expon = exponents[3]
        else:
            addtl_mixing_const = None
            addtl_mixing_expon = 0
        species_dict = {s.name: s for s in dbf.species}
        if addtl_mixing_const is not None:
            additional_mixing_constituent = species_dict.get(addtl_mixing_const.upper(), v.Species(addtl_mixing_const))
        else:
            additional_mixing_constituent = v.Species(None)

        dbf.add_parameter(
            "MQMX", phase_name, constituent_array, param_order=None, param=expr,
            mixing_code=mixing_code, exponents=exponents,
            additional_mixing_constituent=additional_mixing_constituent,
            additional_mixing_exponent=addtl_mixing_expon,
            force_insert=False,
            )


def _species(el_chg):
    el, chg = el_chg
    name = rename_element_charge(el, chg)
    constituents = dict(parse_chemical_formula(el)[0])
    return v.Species(name, constituents=constituents, charge=chg)


def _process_chemical_group_override_string(s):
    """
    Parse strings for special MQMQA (SUBG/SUBQ) parameters that indicate Kohler/Toop
    mixing special cases that are not expressed in the specified chemical groups
    ("chemical group overrides").

    Examples
    --------
    >>> overrides = _process_chemical_group_override_string('3 1 4T3 3 1K 3 4T4 1 4 5')
    >>> assert overrides['ternary_element_indices'] == [3, 1, 4]
    >>> assert [bx['interaction_type'] for bx in overrides['binary_interactions']] == ['T', 'K', 'T']
    >>> assert [bx.get('toop_element_index') for bx in overrides['binary_interactions']] == [3, None, 4]
    >>> assert [bx['interacting_element_indices'] for bx in overrides['binary_interactions']] == [[3, 1], [3, 4], [1, 4]]
    >>> assert overrides['non_mixing_element_index'] == 5

    """
    override_tokens = TokenParser(s.upper().replace("K", " K").replace("T", " T "))
    override_dict = {}
    ternary_element_indices = override_tokens.parseN(3, int)
    override_dict["ternary_element_indices"] = ternary_element_indices
    override_dict["binary_interactions"] = []
    for _ in range(3):
        # Parse the extrapolation type of each binary interaction in the ternary,
        # with the appropriate metadata according to the interaction type
        binary_interaction_dict = {}
        interaction_type = override_tokens.parse(str)
        binary_interaction_dict["interaction_type"] = interaction_type
        if interaction_type == "K":
            # Kohler, no special handling
            pass
        elif interaction_type == "T":
            # Toop, parse which element is the odd-element-out
            binary_interaction_dict["toop_element_index"] = override_tokens.parse(int)
        else:
            raise ValueError(f"Unknown extrapolation type {interaction_type} encountered while processing override string.")
        binary_interaction_dict["interacting_element_indices"] = override_tokens.parseN(2, int)
        override_dict["binary_interactions"].append(binary_interaction_dict)
    override_dict["non_mixing_element_index"] = override_tokens.parse(int)
    return override_dict


@dataclass
class Phase_SUBQ(PhaseBase):
    num_pairs: int
    num_quadruplets: int
    num_subl_1_const: int
    num_subl_2_const: int
    subl_1_const: List[str]
    subl_2_const: List[str]
    subl_1_charges: List[float]
    subl_1_chemical_groups: List[int]
    subl_2_charges: List[float]
    subl_2_chemical_groups: List[int]
    subl_const_idx_pairs: List[Tuple[int, int]]
    quadruplets: List[SUBQQuadrupletCoordinations]
    excess_parameters: List[SUBQExcessQuadruplet]
    chemical_group_overrides: List[str]

    def insert(self, dbf: Database, pure_elements: List[str], gibbs_coefficient_idxs: List[int], excess_coefficient_idxs: List[int]):
        # First: get the pair and quadruplet species added to the database:
        # Here we rename the species names according to their charges, to avoid creating duplicate pairs/quadruplets
        cation_el_chg_pairs = list(zip(self.subl_1_const, self.subl_1_charges))
        # anion charges are given as positive values, but should be negative in
        # order to make the species entered in the expected way (`CL-1`).
        anion_el_chg_pairs = list(zip(self.subl_2_const, [-1*c for c in self.subl_2_charges]))
        # pycalphad usually uses Species to differentiate charged species, for
        # example Species('CU', charge=1) vs. Species('CU', charge=2). In the
        # implementation of the model, we will use Speices to refer to a
        # quadruplet which may have both CU+1 and CU+2 species inside, so we
        # need to add an additional qualifier that mangles the name of the
        # elements in the species so that we can clearly differentiate the
        # (CU+1 CU+1):(XY) quadruplet from the (CU+2 CU+2):(XY) quadruplet. We
        # do that by adding the charge to the name (without +/-), i.e. CU_1 or
        # CU_1.0. This mangling frees us from worrying about Species name
        # collisions within the Database.
        cations = [rename_element_charge(el, chg) for el, chg in cation_el_chg_pairs]
        anions = [rename_element_charge(el, chg) for el, chg in anion_el_chg_pairs]
        # Add the (renamed) species to the database so the phase constituents can be added
        dbf.species.update(map(_species, cation_el_chg_pairs))
        dbf.species.update(map(_species, anion_el_chg_pairs))

        # Second: add the phase and phase constituents
        model_hints = {
            "mqmqa": {
                "type": self.phase_type,
                "chemical_groups": {
                    "cations": dict(zip(map(_species, cation_el_chg_pairs), self.subl_1_chemical_groups)),
                    "anions": dict(zip(map(_species, anion_el_chg_pairs), self.subl_2_chemical_groups)),
                }
            }
        }
        dbf.add_phase(self.phase_name, model_hints, sublattices=[1.0])
        dbf.add_structure_entry(self.phase_name, self.phase_name)
        dbf.add_phase_constituents(self.phase_name, [cations, anions])

        # Third: add the endmember (pair) Gibbs energies
        num_pairs = len(list(itertools.product(cations, anions)))
        assert len(self.endmembers) == num_pairs

        # Endmember pairs came in order of the specified subl_const_idx_pairs labels.
        for (i, j), endmember in zip(self.subl_const_idx_pairs, self.endmembers):
            endmember.insert(dbf, self.phase_name, [[cations[i-1]], [anions[j-1]]], gibbs_coefficient_idxs)

        # Fourth: add parameters for coordinations
        for quadruplet in self.quadruplets:
            quadruplet.insert(dbf, self.phase_name, cations, anions)

        # Fifth: add excess parameters
        for excess_param in self.excess_parameters:
            excess_param.insert(dbf, self.phase_name, cations, anions, excess_coefficient_idxs)

        # Process chemical group overrides - for now we simply warn with the affected species if any overrides are detected.
        for override_string in self.chemical_group_overrides:
            override_dict = _process_chemical_group_override_string(override_string)
            overriden_species_indices = override_dict["ternary_element_indices"] + [override_dict["non_mixing_element_index"]]
            dummy_xs = ExcessBase(overriden_species_indices)
            override_constituents = []
            for subl_constituents in dummy_xs.constituent_array([self.subl_1_const, self.subl_2_const]):
                override_constituents.extend(subl_constituents)
            warnings.warn(f"Phase {self.phase_name} overrides the ternary extrapolation models for the system {override_constituents}. Use caution as extrapolated energies may be incorrect.")


# TODO: not yet supported
@dataclass
class Phase_RealGas(PhaseBase):
    endmembers: List[EndmemberRealGas]


# TODO: not yet supported
@dataclass
class Phase_Aqueous(PhaseBase):
    endmembers: List[EndmemberAqueous]


def tokenize(instring, startline=0, force_upper=False):
    if force_upper:
        return TokenParser('\n'.join(instring.upper().splitlines()[startline:]))
    else:
        return TokenParser('\n'.join(instring.splitlines()[startline:]))


def parse_header(toks: TokenParser) -> Header:
    num_pure_elements = toks.parse(int)
    num_soln_phases = toks.parse(int)
    list_soln_species_count = toks.parseN(num_soln_phases, int)
    num_stoich_phases = toks.parse(int)
    pure_elements = toks.parseN(num_pure_elements, str)
    pure_elements_mass = toks.parseN(num_pure_elements, float)
    num_gibbs_coeffs = toks.parse(int)
    gibbs_coefficient_idxs = toks.parseN(num_gibbs_coeffs, int)
    num_excess_coeffs = toks.parse(int)
    excess_coefficient_idxs = toks.parseN(num_excess_coeffs, int)
    header = Header(list_soln_species_count, num_stoich_phases, pure_elements, pure_elements_mass, gibbs_coefficient_idxs, excess_coefficient_idxs)
    return header


def parse_additional_terms(toks: TokenParser) -> List[AdditionalCoefficientPair]:
    num_additional_terms = toks.parse(int)
    return [AdditionalCoefficientPair(*toks.parseN(2, float)) for _ in range(num_additional_terms)]


def parse_PTVm_terms(toks: TokenParser) -> PTVmTerms:
    # parse molar volume terms, there seem to always be 11 terms (at least in the one file I have)
    return PTVmTerms(toks.parseN(11, float))


def parse_interval_Gibbs(toks: TokenParser, num_gibbs_coeffs, has_additional_terms, has_PTVm_terms) -> IntervalG:
    temperature_max = toks.parse(float)
    coefficients = toks.parseN(num_gibbs_coeffs, float)
    additional_coeff_pairs = parse_additional_terms(toks) if has_additional_terms else []
    PTVm_terms = parse_PTVm_terms(toks) if has_PTVm_terms else []
    return IntervalG(temperature_max, coefficients, additional_coeff_pairs, PTVm_terms)


def parse_interval_heat_capacity(toks: TokenParser, num_gibbs_coeffs, H298, S298, has_H_trans, has_additional_terms, has_PTVm_terms) -> IntervalCP:
    # 6 coefficients are required
    assert num_gibbs_coeffs == 6
    if has_H_trans:
        H_trans = toks.parse(float)
    else:
        H_trans = 0.0  # 0.0 will be added to the first (or only) interval
    temperature_max = toks.parse(float)
    CP_coefficients = toks.parseN(4, float)
    additional_coeff_pairs = parse_additional_terms(toks) if has_additional_terms else []
    PTVm_terms = parse_PTVm_terms(toks) if has_PTVm_terms else []
    return IntervalCP(temperature_max, H298, S298, CP_coefficients, H_trans, additional_coeff_pairs, PTVm_terms)


def parse_endmember(toks: TokenParser, num_pure_elements, num_gibbs_coeffs, is_stoichiometric=False):
    species_name = toks.parse(str)
    if toks[0] == '#':
        # special case for stoichiometric phases, this is a dummy species, skip it
        _ = toks.parse(str)
    try:
        gibbs_eq_type = toks.parse(int)
    except TokenParserError:
        # There may be two floats that come after the species name on the same
        # line. The meaning is not yet clear, but they are often zero. If they
        # are zero, we will throw them away. This drops into some private APIs
        # for TokenParser until there's another way to handle these values.
        f1 = toks.parse(float)
        if f1 != 0.0:
            raise TokenParserError(f"Non-zero values are not yet supported after species {species_name}. Got {f1} at line number {toks._line_number + 1} for line:\n    {toks._current_line}")
        f2 = toks.parse(float)
        if f2 != 0.0:
            raise TokenParserError(f"Non-zero values are not yet supported after species {species_name}. Got {f2} at line number {toks._line_number + 1} for line:\n    {toks._current_line}")
        gibbs_eq_type = toks.parse(int)
    # Determine how to parse the type of thermodynamic option
    has_magnetic = gibbs_eq_type > 12
    gibbs_eq_type_reduced = (gibbs_eq_type - 12) if has_magnetic else gibbs_eq_type
    is_gibbs_energy_interval = gibbs_eq_type_reduced in (1, 2, 3, 4, 5, 6)
    is_heat_capacity_interval = gibbs_eq_type_reduced in (7, 8, 9, 10, 11, 12)
    has_additional_terms = gibbs_eq_type_reduced in (4, 5, 6, 10, 11, 12)
    has_constant_Vm_terms = gibbs_eq_type_reduced in (2, 5, 8, 11)
    has_PTVm_terms = gibbs_eq_type_reduced in (3, 6, 9, 12)
    num_intervals = toks.parse(int)
    stoichiometry_pure_elements = toks.parseN(num_pure_elements, float)
    if has_constant_Vm_terms:
        raise ValueError("Constant molar volume equations (thermodynamic data options (2, 5, 8, 11)) are not supported yet.")
    # Piecewise endmember energy intervals
    if is_gibbs_energy_interval:
        intervals = [parse_interval_Gibbs(toks, num_gibbs_coeffs, has_additional_terms, has_PTVm_terms) for _ in range(num_intervals)]
    elif is_heat_capacity_interval:
        H298 = toks.parse(float)
        S298 = toks.parse(float)
        # parse the first without H_trans, then parse the rest
        intervals = [parse_interval_heat_capacity(toks, num_gibbs_coeffs, H298, S298, False, has_additional_terms, has_PTVm_terms)]
        for _ in range(num_intervals - 1):
            intervals.append(parse_interval_heat_capacity(toks, num_gibbs_coeffs, H298, S298, True, has_additional_terms, has_PTVm_terms))
    else:
        raise ValueError(f"Unknown thermodynamic data option type {gibbs_eq_type}. A number in [1, 24].")
    # magnetic terms
    if has_magnetic:
        curie_temperature = toks.parse(float)
        magnetic_moment = toks.parse(float)
        return EndmemberMagnetic(species_name, gibbs_eq_type, stoichiometry_pure_elements, intervals, curie_temperature, magnetic_moment)
    return Endmember(species_name, gibbs_eq_type, stoichiometry_pure_elements, intervals)


def parse_endmember_qkto(toks: TokenParser, num_pure_elements: int, num_gibbs_coeffs: int):
    # add an extra "pure element" to parse the charge
    em = parse_endmember(toks, num_pure_elements, num_gibbs_coeffs)
    stoichiometric_factor = toks.parse(float)
    chemical_group = toks.parse(int)
    return EndmemberQKTO(em.species_name, em.gibbs_eq_type, em.stoichiometry_pure_elements, em.intervals, stoichiometric_factor, chemical_group)


def parse_endmember_aqueous(toks: TokenParser, num_pure_elements: int, num_gibbs_coeffs: int):
    # add an extra "pure element" to parse the charge
    em = parse_endmember(toks, num_pure_elements + 1, num_gibbs_coeffs)
    return EndmemberAqueous(em.species_name, em.gibbs_eq_type, em.stoichiometry_pure_elements[1:], em.intervals, em.stoichiometry_pure_elements[0])


def parse_endmember_subq(toks: TokenParser, num_pure_elements, num_gibbs_coeffs, zeta=None):
    em = parse_endmember(toks, num_pure_elements, num_gibbs_coeffs)
    # The first two entries are the stoichiometry of the pair (the cation and anion). It's unclear to what the last three are for.
    stoichiometry_quadruplet = toks.parseN(5, float)
    if zeta is None:
        # This is SUBQ we need to parse it. If zeta is passed, that means we're in SUBG mode
        zeta = toks.parse(float)
    return SUBQPair(em.species_name, em.gibbs_eq_type, em.stoichiometry_pure_elements, em.intervals, stoichiometry_quadruplet, zeta)


def parse_quadruplet(toks):
    quad_idx = toks.parseN(4, int)
    quad_coords = toks.parseN(4, float)
    return SUBQQuadrupletCoordinations(quad_idx, quad_coords)


def parse_subq_excess(toks, mixing_type, num_excess_coeffs):
    mixing_code = toks.parse(str)
    mixing_const = toks.parseN(4, int)
    mixing_exponents = toks.parseN(4, int)
    metadata = toks.parseN(12, float)  # TODO: not sure what this metadata is - could it be more parameters? They are usually all zeros.
    additional_cation_mixing_const = toks.parse(int)
    additional_anion_mixing_exponent = toks.parse(int)
    excess_coeffs = toks.parseN(num_excess_coeffs, float)
    return SUBQExcessQuadruplet(mixing_type, mixing_code, mixing_const, mixing_exponents, metadata, additional_cation_mixing_const, additional_anion_mixing_exponent, excess_coeffs)


def parse_phase_subq(toks, phase_name, phase_type, num_pure_elements, num_gibbs_coeffs, num_excess_coeffs):
    if phase_type == 'SUBG':
        zeta = toks.parse(float)
    else:
        zeta = None
    num_pairs = toks.parse(int)
    num_quadruplets = toks.parse(int)
    endmembers = [parse_endmember_subq(toks, num_pure_elements, num_gibbs_coeffs, zeta=zeta) for _ in range(num_pairs)]
    num_subl_1_const = toks.parse(int)
    num_subl_2_const = toks.parse(int)
    subl_1_const = toks.parseN(num_subl_1_const, str)
    subl_2_const = toks.parseN(num_subl_2_const, str)
    subl_1_charges = toks.parseN(num_subl_1_const, float)
    subl_1_chemical_groups = toks.parseN(num_subl_1_const, int)
    subl_2_charges = toks.parseN(num_subl_2_const, float)
    subl_2_chemical_groups = toks.parseN(num_subl_2_const, int)
    subl_1_pair_idx = toks.parseN(num_subl_1_const*num_subl_2_const, int)
    subl_2_pair_idx = toks.parseN(num_subl_1_const*num_subl_2_const, int)
    subl_const_idx_pairs = [(s1i, s2i) for s1i, s2i in zip(subl_1_pair_idx, subl_2_pair_idx)]
    quadruplets = [parse_quadruplet(toks) for _ in range(num_quadruplets)]
    excess_parameters = []
    chemical_group_overrides = []
    while True:
        mixing_type = toks.parse(int)
        if mixing_type == 0:
            break
        elif mixing_type < 0:
            # For mixing type -N, there are N*10 tokens.
            # The tokens look something like `1 2 3K 1 2K 1 3K 2 3 6`
            for _ in range(-mixing_type):
                # For each entry, simply parse the whole string
                chemical_group_overrides.append(" ".join(toks.parseN(10, str)))
            break
        excess_parameters.append(parse_subq_excess(toks, mixing_type, num_excess_coeffs))

    return Phase_SUBQ(phase_name, phase_type, endmembers, num_pairs, num_quadruplets, num_subl_1_const, num_subl_2_const, subl_1_const, subl_2_const, subl_1_charges, subl_1_chemical_groups, subl_2_charges, subl_2_chemical_groups, subl_const_idx_pairs, quadruplets, excess_parameters, chemical_group_overrides)


def parse_excess_magnetic_parameters(toks):
    excess_terms = []
    while True:
        num_interacting_species = toks.parse(int)
        if num_interacting_species == 0:
            break
        interacting_species_idxs = toks.parseN(num_interacting_species, int)
        num_terms = toks.parse(int)
        for parameter_order in range(num_terms):
            curie_temperature = toks.parse(float)
            magnetic_moment = toks.parse(float)
            excess_terms.append(ExcessRKMMagnetic(interacting_species_idxs, parameter_order, curie_temperature, magnetic_moment))
    return excess_terms


def parse_excess_parameters(toks, num_excess_coeffs):
    excess_terms = []
    while True:
        num_interacting_species = toks.parse(int)
        if num_interacting_species == 0:
            break
        interacting_species_idxs = toks.parseN(num_interacting_species, int)
        num_terms = toks.parse(int)
        for parameter_order in range(num_terms):
            excess_terms.append(ExcessRKM(interacting_species_idxs, parameter_order, toks.parseN(num_excess_coeffs, float)))
    return excess_terms


def parse_excess_parameters_pitz(toks, num_excess_coeffs):
    excess_terms = []
    while True:
        num_interacting_species = toks.parse(int)
        if num_interacting_species == 0:
            break
        # there are always 3 ints, regardless of the above "number of interacting species", if the number of interactings species is 2, we'll just throw the third number away for now
        if num_interacting_species == 2:
            interacting_species_idxs = toks.parseN(num_interacting_species, int)
            toks.parse(int)
        elif num_interacting_species == 3:
            interacting_species_idxs = toks.parseN(num_interacting_species, int)
        else:
            raise ValueError(f"Invalid number of interacting species for Pitzer model, got {num_interacting_species} (expected 2 or 3).")
        parameter_order = None
        excess_terms.append(ExcessRKM(interacting_species_idxs, parameter_order, toks.parseN(num_excess_coeffs, float)))
    return excess_terms


def parse_excess_qkto(toks, num_excess_coeffs):
    excess_terms = []
    while True:
        num_interacting_species = toks.parse(int)
        if num_interacting_species == 0:
            break
        interacting_species_idxs = toks.parseN(num_interacting_species, int)
        exponents = toks.parseN(num_interacting_species, int)
        coefficients = toks.parseN(num_excess_coeffs, float)
        excess_terms.append(ExcessQKTO(interacting_species_idxs, exponents, coefficients))
    return excess_terms


def parse_phase_cef(toks, phase_name, phase_type, num_pure_elements, num_gibbs_coeffs, num_excess_coeffs, num_const):
    is_magnetic_phase_type = len(phase_type) == 5 and phase_type[4] == 'M'
    sanitized_phase_type = phase_type[:4]  # drop the magnetic contribution, only used for matching later (i.e. SUBLM -> SUBL)
    if is_magnetic_phase_type:
        allowed_magnetic_phase_types = ('RKMP', 'QKTO', 'SUBL', 'SUBM', 'SUBG', 'SUBQ', 'WAGN')
        if sanitized_phase_type not in allowed_magnetic_phase_types:
            raise ValueError(f'Magnetic phase type {phase_type} is only supported for {allowed_magnetic_phase_types}')
        magnetic_afm_factor = toks.parse(float)
        magnetic_structure_factor = toks.parse(float)
    else:
        magnetic_afm_factor = None
        magnetic_structure_factor = None

    endmembers = []
    for _ in range(num_const):
        if sanitized_phase_type == 'PITZ':
            endmembers.append(parse_endmember_aqueous(toks, num_pure_elements, num_gibbs_coeffs))
        elif sanitized_phase_type == 'QKTO':
            endmembers.append(parse_endmember_qkto(toks, num_pure_elements, num_gibbs_coeffs))
        else:
            endmembers.append(parse_endmember(toks, num_pure_elements, num_gibbs_coeffs))

    # defining sublattice model
    if sanitized_phase_type in ('SUBL',):
        num_subl = toks.parse(int)
        subl_atom_fracs = toks.parseN(num_subl, float)
        # some phases have number of atoms after a colon in the phase name, e.g. SIGMA:30
        if len(phase_name.split(':')) > 1:
            num_atoms = float(phase_name.split(':')[1])
        else:
            num_atoms = 1.0
        subl_ratios = [num_atoms*subl_frac for subl_frac in subl_atom_fracs]
        # read the data used to recover the mass, it's redundant and doesn't need to be stored
        subl_constituents = toks.parseN(num_subl, int)
        constituent_array = []
        for num_subl_species in subl_constituents:
            constituent_array.append(toks.parseN(num_subl_species, str))
        num_endmembers = int(np.prod(subl_constituents))
        endmember_constituent_idxs = []
        for _ in range(num_subl):
            endmember_constituent_idxs.append(toks.parseN(num_endmembers, int))
        # endmember_constituents now is like [[1, 2, 3, 4], [1, 1, 1, 1]] for
        # a two sublattice phase with (4, 1) constituents
        # we want to invert this so each endmember is a pair, i.e.
        # endmember_constituents = [[1, 1], [2, 1], [3, 1], [4, 1]]
        endmember_constituent_idxs = list(zip(*endmember_constituent_idxs))
    elif sanitized_phase_type in ('IDMX', 'RKMP', 'QKTO', 'PITZ'):
        subl_ratios = [1.0]
        constituent_array = None  # implictly defined by the endmembers
        endmember_constituent_idxs = None
    else:
        raise NotImplementedError(f"Phase type {phase_type} does not have method defined for determing the sublattice ratios")

    # excess terms
    if sanitized_phase_type in ('IDMX',):
        # No excess parameters
        excess_parameters = []
    elif sanitized_phase_type in ('PITZ',):
        excess_parameters = parse_excess_parameters_pitz(toks, num_excess_coeffs)
    elif sanitized_phase_type in ('RKMP', 'SUBL'):
        # SUBL will have no excess parameters, but it will have the "0" terminator like it has excess parameters so we can use the excess parameter parsing to process it all the same.
        excess_parameters = []
        if is_magnetic_phase_type:
            excess_parameters.extend(parse_excess_magnetic_parameters(toks))
        excess_parameters.extend(parse_excess_parameters(toks, num_excess_coeffs))
    elif sanitized_phase_type in ('QKTO',):
        excess_parameters = parse_excess_qkto(toks, num_excess_coeffs)
    return Phase_CEF(phase_name, phase_type, endmembers, subl_ratios, constituent_array, endmember_constituent_idxs, excess_parameters, magnetic_afm_factor=magnetic_afm_factor, magnetic_structure_factor=magnetic_structure_factor)


def parse_phase_real_gas(toks, phase_name, phase_type, num_pure_elements, num_gibbs_coeffs, num_const):
    endmembers = []
    for _ in range(num_const):
        em = parse_endmember(toks, num_pure_elements, num_gibbs_coeffs)
        Tc = toks.parse(float)
        Pc = toks.parse(float)
        Vc = toks.parse(float)
        acentric_factor = toks.parse(float)
        dipole_moment = toks.parse(float)
        endmembers.append(EndmemberRealGas(em.species_name, em.gibbs_eq_type, em.stoichiometry_pure_elements, em.intervals, Tc, Pc, Vc, acentric_factor, dipole_moment))
    return Phase_RealGas(phase_name, phase_type, endmembers)


def parse_phase_aqueous(toks, phase_name, phase_type, num_pure_elements, num_gibbs_coeffs, num_const):
    endmembers = [parse_endmember_aqueous(toks, num_pure_elements, num_gibbs_coeffs) for _ in range(num_const)]
    return Phase_Aqueous(phase_name, phase_type, endmembers)


def parse_phase(toks, num_pure_elements, num_gibbs_coeffs, num_excess_coeffs, num_const):
    """Dispatches to the correct parser depending on the phase type"""
    phase_name = toks.parse(str)
    phase_type = toks.parse(str)
    if phase_type in ('SUBQ', 'SUBG'):
        phase = parse_phase_subq(toks, phase_name, phase_type, num_pure_elements, num_gibbs_coeffs, num_excess_coeffs)
    elif phase_type == 'IDVD':
        phase = parse_phase_real_gas(toks, phase_name, phase_type, num_pure_elements, num_gibbs_coeffs, num_const)
    elif phase_type == 'IDWZ':
        phase = parse_phase_aqueous(toks, phase_name, phase_type, num_pure_elements, num_gibbs_coeffs, num_const)
    elif phase_type in ('IDMX', 'RKMP', 'RKMPM', 'QKTO', 'SUBL', 'SUBLM', 'PITZ'):
        # all these phases parse the same
        phase = parse_phase_cef(toks, phase_name, phase_type, num_pure_elements, num_gibbs_coeffs, num_excess_coeffs, num_const)
    else:
        raise NotImplementedError(f"phase type {phase_type} not yet supported")
    return phase


def parse_stoich_phase(toks, num_pure_elements, num_gibbs_coeffs):
    endmember = parse_endmember(toks, num_pure_elements, num_gibbs_coeffs, is_stoichiometric=True)
    if isinstance(endmember, EndmemberMagnetic):
        magnetic_afm_factor = toks.parse(float)
        magnetic_structure_factor = toks.parse(float)
    else:
        magnetic_afm_factor = None
        magnetic_structure_factor = None
    phase_name = endmember.species_name
    return Phase_Stoichiometric(phase_name, None, [endmember], magnetic_afm_factor=magnetic_afm_factor, magnetic_structure_factor=magnetic_structure_factor)


def parse_cs_dat(instring):
    toks = tokenize(instring, startline=1)
    header = parse_header(toks)
    num_pure_elements = len(header.pure_elements)
    num_gibbs_coeffs = len(header.gibbs_coefficient_idxs)
    num_excess_coeffs = len(header.excess_coefficient_idxs)
    # num_const = 0 is gas phase that isn't present, so skip it
    solution_phases = [parse_phase(toks, num_pure_elements, num_gibbs_coeffs, num_excess_coeffs, num_const) for num_const in header.list_soln_species_count if num_const != 0]
    stoichiometric_phases = [parse_stoich_phase(toks, num_pure_elements, num_gibbs_coeffs) for _ in range(header.num_stoich_phases)]
    # By convention there are sometimes comments at the end that we ignore.
    # Any remaining lines after the number of prescribed phases and
    # stoichiometric compounds are not parsed.
    return header, solution_phases, stoichiometric_phases


def read_cs_dat(dbf: Database, fd):
    """
    Parse a ChemSage DAT file into a pycalphad Database object.

    Parameters
    ----------
    dbf : Database
        A pycalphad Database.
    fd : file-like
        File descriptor.
    """
    header, solution_phases, stoichiometric_phases = parse_cs_dat(fd.read().upper())
    # add elements and their reference states
    for el, mass in zip(header.pure_elements, header.pure_elements_mass):
        if 'E(' not in str(el):
            dbf.elements.add(el)
            dbf.species.add(v.Species(el))
            # add element reference state data
            dbf.refstates[el] = {
                'mass': mass,
                # the following metadata is not given in DAT files,
                # but is standard for our Database files
                'phase': 'BLANK',
                'H298': 0.0,
                'S298': 0.0,
            }
    # Each phase subclass knows how to insert itself into the database.
    # The insert method will appropriately insert all endmembers as well.
    processed_phases = []
    for parsed_phase in (*solution_phases, *stoichiometric_phases):
        if parsed_phase.phase_name in processed_phases:
            # DAT files allow multiple entries of the same phase to handle
            # miscibility gaps. We discard the duplicate phase definitions.
            continue
        parsed_phase.insert(dbf, header.pure_elements, header.gibbs_coefficient_idxs, header.excess_coefficient_idxs)
        processed_phases.append(parsed_phase.phase_name)

    # process all the parameters that got added with dbf.add_parameter
    dbf.process_parameter_queue()

Database.register_format("dat", read=read_cs_dat, write=None)
