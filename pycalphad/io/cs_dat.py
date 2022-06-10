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
import datetime

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


def reflow_text(text, linewidth=80):
    """
    Add line breaks to ensure text doesn't exceed a certain line width.

    Parameters
    ----------
    text : str
    linewidth : int, optional

    Returns
    -------
    reflowed_text : str
    """
    lines = text.split("\n")
    linebreak_chars = [" "]
    output_lines = []
    line_counter = 0
    for line in lines:
        # Don't break lines below set width, or first (comment) line of DAT
        if len(line) <= linewidth or line_counter == 0:
            output_lines.append(line.rstrip())
        else:
            while len(line) > linewidth:
                linebreak_idx = linewidth - 1
                while linebreak_idx > 0 and line[linebreak_idx] not in linebreak_chars:
                    linebreak_idx -= 1
                # Need to check 2 (rather than zero) because we prepend newlines with 2 characters
                if linebreak_idx <= 2:
                    raise ValueError(f"Unable to reflow the following line of length {len(line)} below the maximum length of {linewidth}: \n{line}")
                output_lines.append(line[:linebreak_idx].rstrip())
                line = line[linebreak_idx:]
            output_lines.append(line.rstrip())
        line_counter += 1
    # CRLF for FactSage compatibility
    return "\r\n".join(output_lines)


atomic_number_map = [
    'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P',
    'S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
    'Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh',
    'Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd',
    'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re',
    'Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th',
    'Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db',
    'Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts', 'Og'
]


def write_cs_dat(dbf: Database, fd, if_incompatible='warn'):
    """
    Write a DAT file from a pycalphad Database object.

    The goal is to produce DATs that conform to the most restrictive subset of database specifications. FactSage requires
    fixed value widths and a maximum line length of 80 characters. FactSage 8.0 (and earlier) format will be produced.
    The default is to warn the user when attempting to write an incompatible database and the user must choose whether to
    warn and write the file anyway or to fix the incompatibility.

    Other DAT compatibility issues required by FactSage or other software should be reported to the issue tracker.

    Parameters
    ----------
    dbf : Database
        A pycalphad Database.
    fd : file-like
        File descriptor.
    if_incompatible : string, optional ['raise', 'warn', 'fix']
        Strategy if the database does not conform to the most restrictive database specification.
        The 'warn' option (default) will write out the incompatible database with a warning.
        The 'raise' option will raise a DatabaseExportError.
        The 'ignore' option will write out the incompatible database silently.
        The 'fix' option will rectify the incompatibilities e.g. through name mangling.
    """
    # Needed for database queries
    from tinydb import where
    # Before writing anything, check that the TDB is valid and take the appropriate action if not
    if if_incompatible not in ['warn', 'raise', 'ignore', 'fix']:
        raise ValueError('Incorrect options passed to \'if_invalid\'. Valid args are \'raise\', \'warn\', or \'fix\'.')

    # Begin constructing the written database
    writetime = datetime.datetime.now()
    maxlen = 80
    output = ""
    # Comment header block
    # Import here to prevent circular imports
    from pycalphad import __version__
    try:
        # getuser() will raise on Windows if it can't find a username: https://bugs.python.org/issue32731
        username = getpass.getuser()
    except:
        # if we can't find a good username, just choose a default and move on
        username = 'user'
    output += " Date: {} ".format(writetime.strftime("%Y-%m-%d %H:%M"))

    # DAT standard is elements written from highest atomic number to lowest
    elements = list(dbf.elements)
    element_order = np.argsort([atomic_number_map.index(el.capitalize())+1 for el in elements])[::-1]
    elements_ordered = [elements[i] for i in element_order]
    output += "System: {} ".format('-'.join([elements[i] for i in element_order]))
    output += "| Generated by {} (pycalphad {})\n".format(username, __version__)

    # Make a list of phase types (models) to support for writing DAT
    supported_phase_types = ['IDMX','QKTO','RKMP','SUBL','RKMPM','SUBLM','SUBG','SUBQ'] # TODO: 'SUBI'

    # Get numbers of solution phases (and type/species for each) and pure condensed phases
    solution_phases = []
    stoichiometric_phases = []
    solution_phase_types = []
    solution_phase_species = []

    # DAT *always* includes ideal gas phase (gas_ideal) in header (in first solution phase position), even if not used
    solution_phases.insert(0,'GAS_IDEAL')
    solution_phase_types.insert(0,'IDMX')
    # If there isn't really an ideal gas phase, need empty list of species, so just insert this right away
    solution_phase_species.insert(0,[])

    # Loop over phases and find stoichiometric and supported solution phases
    # TODO: append phase (and info) again for miscibility gaps (from an argument?)
    # Also this gets the species names but only really needs a count at this point
    for phase_name in dbf.phases:
        # print(phase_name)
        # print(dbf.phases[phase_name])
        # If all sublattices are singly occupied, it is a stoichiometric phase
        if all([len(subl) == 1 for subl in dbf.phases[phase_name].constituents]):
            stoichiometric_phases.append(phase_name)
        else:
            # Check if an ideal gas phase
            if phase_name.upper() == 'GAS_IDEAL':
                # Replace blank species list in first position with actual ideal gas species
                solution_phase_species[0] = [[i.name for i in constituent] for constituent in dbf.phases[phase_name].constituents][0]
                continue
            # Check if a MQMQA phase
            if dbf.phases[phase_name].model_hints:
                try:
                    type = dbf.phases[phase_name].model_hints['mqmqa']['type']
                except KeyError:
                    # Not MQMQA, and that's ok
                    pass
                else:
                    # This is an MQMQA-type phase
                    solution_phases.append(phase_name)
                    # Save MQMQA sub-type (SUBG or SUBQ)
                    solution_phase_types.append(type)
                    # Determine species for phase
                    constituents = [[i.name for i in constituent] for constituent in dbf.phases[phase_name].constituents]
                    # Species will be quadruplets for counting purposes
                    species = []
                    for i in range(len(constituents[0])):
                        for j in range(i,len(constituents[0])):
                            for k in range(len(constituents[1])):
                                for l in range(k,len(constituents[1])):
                                    species.append(f'{constituents[0][i]},{constituents[0][j]}/{constituents[1][k]},{constituents[1][l]}')
                    solution_phase_species.append(species)
                    continue
            # Check if QKTO: if phase has any "QKT" parameters, it is QKTO
            detect_query = (
                (where("phase_name") == phase_name) & \
                (where("parameter_type") == "QKT")
            )
            params = list(dbf._parameters.search(detect_query))
            if len(params) > 0:
                # This phase is QKTO
                solution_phases.append(phase_name)
                solution_phase_types.append("QKTO")
                species = [[i.name for i in constituent] for constituent in dbf.phases[phase_name].constituents][0]
                solution_phase_species.append(species)
                continue

            # Everything else is a CEF variant
            nSublattices = len(dbf.phases[phase_name].sublattices)
            if nSublattices > 1:
                # If multiple sublattices, identify type
                solution_phases.append(phase_name)
                solution_phase_types.append("SUBL")
                constituents = [[i.name for i in constituent] for constituent in dbf.phases[phase_name].constituents]
                # Make species from products of constituents
                species = [':'.join([con.capitalize() for con in end]) for end in list(itertools.product(*constituents))]
                solution_phase_species.append(species)
            else:
                # Otherwise RKMP phase
                solution_phases.append(phase_name)
                solution_phase_types.append("RKMP")
                species = [[i.name for i in constituent] for constituent in dbf.phases[phase_name].constituents][0]
                solution_phase_species.append(species)

            # Check if magnetic
            if dbf.phases[phase_name].model_hints:
                if 'ihj_magnetic_structure_factor' in dbf.phases[phase_name].model_hints:
                    solution_phase_types[-1] += ('M')


    # Number of elements, phases, species line
    solution_phase_species_counts = ' '.join([f'{len(species):4}' for species in solution_phase_species])
    output += f" {len(dbf.elements):4} {len(solution_phases):4} {solution_phase_species_counts} {len(stoichiometric_phases):4}"

    # List of elements lines
    for i in range(len(elements)):
        if np.mod(i,3) == 0:
            output += "\n"
        output += f" {elements[element_order[i]].capitalize():24}"

    # Element masses lines
    for i in range(len(elements)):
        if np.mod(i,3) == 0:
            output += "\n"
            mass = f"{dbf.refstates[elements[element_order[i]]]['mass']:15.8f}"
        else:
            mass = f"{dbf.refstates[elements[element_order[i]]]['mass']:25.8f}"
        output += f"{mass}"
    output += "\n"

    # Two lines to list Gibbs energy parameters used. Hardcoded for now.
    # TODO: Detect these for database.
    # TODO: Implement heat capacity model parameter lists.
    output += '   6   1   2   3   4   5   6\n'
    output += '   6   1   2   3   4   5   6\n'

    # Loop over solution phases and write parameters depending on phase model
    for i in range(len(solution_phases)):
        # Grab info for current phase... a dict might be smarter but would mess up the easiest miscibility gap implementation
        phase_name = solution_phases[i]
        phase_model = solution_phase_types[i]
        phase_species = solution_phase_species[i]
        if len(phase_species) == 0:
            continue
        output += f' {phase_name}\n'
        output += f' {phase_model}\n'

        # Write magnetic parameters for phase
        if phase_model in ('RKMPM', 'SUBLM'):
            ihj_magnetic_structure_factor = dbf.phases[phase_name].model_hints['ihj_magnetic_structure_factor']
            ihj_magnetic_afm_factor = -1/dbf.phases[phase_name].model_hints['ihj_magnetic_afm_factor']
            output += f'  {ihj_magnetic_afm_factor:.5f}     {ihj_magnetic_structure_factor:.5f}\n'

            # Get all magentic parameters for phase
            detect_query = (
                (where("phase_name") == phase_name) & \
                (where("parameter_type") == "TC")
            )
            tcs = dbf._parameters.search(detect_query)

            detect_query = (
                (where("phase_name") == phase_name) & \
                (where("parameter_type") == "BMAGN")
            )
            bmagns = dbf._parameters.search(detect_query)

        # Get endmembers and other parameters depending on phase model
        if phase_model in ('SUBG', 'SUBQ'):
            # Get parameters for endmembers
            detect_query = (
                (where("phase_name") == phase_name) & \
                (where("parameter_type") == "MQMG")
            )
            endmember_params = dbf._parameters.search(detect_query)
            # Write zeta for SUBG
            if phase_model == 'SUBG':
                # All zetas are the same, so grab the first one
                zeta = endmember_params[0]['zeta']
                output += f'{zeta:9.5f}\n'
            # Get parameters for quadruplets
            detect_query = (
                (where("phase_name") == phase_name) & \
                (where("parameter_type") == "MQMZ")
            )
            quadruplet_params = dbf._parameters.search(detect_query)
            # Write number of endmembers and number of non-default coordination sets next
            number_of_endmembers = len(endmember_params)
            number_of_non_default_quadruplets = len(quadruplet_params)
            output += f'{number_of_endmembers:4} {number_of_non_default_quadruplets:3}\n'
        elif phase_model in ('IDMX', 'RKMP', 'RKMPM', 'QKTO', 'SUBL', 'SUBLM', 'PITZ'):
            # Get species for CEF phases
            detect_query = (
                (where("phase_name") == phase_name) & \
                (where("parameter_type") == "G")
            )
            endmember_params = dbf._parameters.search(detect_query)

        # Get sublattice weights
        if phase_model in ('SUBL','SUBLM'):
            sublattice_weights = itertools.cycle(dbf.phases[phase_name].sublattices)
        else:
            sublattice_weights = itertools.cycle([1])

        # Write endmember data
        endmember_names = []
        for endmember in endmember_params:
            # Generate species names and stoichiometries
            name = ''
            stoichiometry = [0 for _ in elements_ordered]
            # Overwrite sublattice_weights for MQM phases
            if phase_model in ('SUBG', 'SUBQ'):
                sublattice_weights = itertools.cycle(endmember['stoichiometry'])
            for speciesList in endmember['constituent_array']:
                for species in speciesList:
                    for element in species.constituents:
                        # Get current sublattice weight
                        weight = next(sublattice_weights)
                        try:
                            stoichiometry[elements_ordered.index(element)] += species.constituents[element] * weight
                        except ValueError:
                            if element.capitalize() != 'Va':
                                print(f'Constituent {element} not found in element list')
                        else:
                            # Add element name to endmember name if not vacancy
                            if element.capitalize() != 'Va':
                                name += element.capitalize()
                                stoich = species.constituents[element] * weight
                                # Add stoichiometric factor to name if not 1
                                if stoich != 1:
                                    if stoich % 1 == 0:
                                        name += f'{int(stoich)}'
                                    else:
                                        name += f'{species.constituents[element]:.2g}'
            output += f' {name}\n'
            endmember_names.append(name)

            # Writeable stoichiometry
            stoichiometry_string = ''.join([f'{stoich:7.1f}' for stoich in stoichiometry])

            # Determine equation type and number of intervals
            gibbs_equation = endmember['parameter'].args
            eq_type, number_of_intervals, gibbs_parameters = parse_gibbs_coefficients_piecewise(gibbs_equation)

            # Write equation type and stoichiometry line
            output += f'{eq_type:4} {number_of_intervals:2}{stoichiometry_string}\n'
            # Write Gibbs parameters line
            output += gibbs_parameters

            # Write stoichiometric factor and chemical group
            # It looks like these aren't actually read/supported currently, so just writing "  1.00000      1" for now
            if phase_model == 'QKTO':
                output += '  1.00000      1\n'
            elif phase_model in ('SUBG', 'SUBQ'):
                stoich_string = '      '.join([f"{n:.5f}" for n in endmember['stoichiometry']])
                output += f'  {stoich_string}\n'

            # Get magnetic parameters for endmember
            if phase_model in ('RKMPM', 'SUBLM'):
                tc_value = 0
                bmagn_value = 0
                caps_name = name.upper()
                # Find tc for endmember
                for tc in tcs:
                    # These will have length 1 constituent arrays, I think (longer are for mixing)
                    if len(tc['constituent_array'][0]) > 1:
                        continue
                    # Check if endmember name matches
                    if str(tc['constituent_array'][0][0]) != caps_name:
                        continue
                    tc_value = tc['parameter']
                    # Delete parameter from array: thus at the end only mixing terms will remain
                    tcs.remove(tc)
                    break
                # Find bmagn for endmember
                for bmagn in bmagns:
                    # These will have length 1 constituent arrays, I think (longer are for mixing)
                    if len(bmagn['constituent_array'][0]) > 1:
                        continue
                    # Check if endmember name matches
                    if str(bmagn['constituent_array'][0][0]) != caps_name:
                        continue
                    bmagn_value = bmagn['parameter']
                    # Delete parameter from array: thus at the end only mixing terms will remain
                    bmagns.remove(bmagn)
                    break
                # Write magnetic parameters line
                output += f' {format_coefficient_mag(tc_value)}{format_coefficient_mag(bmagn_value)}\n'

        # Do constituent mapping for sublattice phases
        if phase_model in ('SUBL','SUBLM','SUBG', 'SUBQ'):
            # Make list of constituents
            if phase_model in ('SUBL','SUBLM'):
                constituents = [[i.name for i in constituent] for constituent in dbf.phases[phase_name].constituents]
            elif phase_model in ('SUBG', 'SUBQ'):
                chemical_groups = dbf.phases[phase_name].model_hints['mqmqa']['chemical_groups']
                constituents = [[species.name for species in chemical_groups[ion].keys()] for ion in ['cations','anions']]
            flat_constituents = [constituent for sublattice in constituents for constituent in sublattice]

            # Get constituent mapping
            constituent_mapping = make_constituent_mapping(constituents, endmember_params)

            # Get sublattice info
            sublattices = dbf.phases[phase_name].sublattices
            nSublattices = len(sublattices)

            # Write sublattice information
            if phase_model in ('SUBL','SUBLM'):
                # Number of sublattices and weights only for SUBL
                output += f'{nSublattices:4}\n'
                output += f'  {"      ".join([f"{weight:.5f}" for weight in sublattices])}\n'
            output += f'{"".join([f"{len(sub):4}" for sub in constituents])}\n'

            # Write constituent names
            for sub in constituents:
                output += f'  {"".join([f"{constituent.capitalize():25}" for constituent in sub])}\n'

            # Write charge magnitudes and chemical groups for MQM
            if phase_model in ('SUBG', 'SUBQ'):
                for ion in ['cations','anions']:
                    charges = [abs(species.charge) for species in chemical_groups[ion].keys()]
                    groups = [chemical_groups[ion][species] for species in chemical_groups[ion].keys()]
                    # Write charges for ion type
                    output += f'  {"      ".join([f"{charges[i]:.5f}" for i in range(len(charges))])}\n'
                    # Write chemical groups for ion type
                    output += f'   {"".join([f"{groups[i]:4}" for i in range(len(charges))])}\n'

            # Write constituent-to-endmember pairing arrays
            for sub in constituent_mapping:
                output += f'{"".join([f"{constituent:4}" for constituent in sub])}\n'

            # Write quadruplet coordination numbers for MQM
            if phase_model in ('SUBG', 'SUBQ'):
                detect_query = (
                    (where("phase_name") == phase_name) & \
                    (where("parameter_type") == "MQMZ")
                )
                params = dbf._parameters.search(detect_query)
                for param in params:
                    con = []
                    for ion in param['constituent_array']:
                        for species in ion:
                            con.append(flat_constituents.index(species.name) + 1)
                    con_string = ''.join([f"{c:4}" for c in con])
                    z_string = '      '.join([f"{z:.7f}" for z in param['coordinations']])
                    output += f'{con_string}  {z_string}\n'

        # Write magnetic excess mixing data
        if phase_model in ('RKMPM', 'SUBLM'):
            # Get excess magnetic parameters
            for tc in tcs:
                tc_value = tc['parameter']
                tc_constituents = tc['constituent_array']
                tcs.remove(tc)
                bmagn_value = 0
                for bmagn in bmagns:
                    # Look for matching bmagn
                    if bmagn['constituent_array'] != tc_constituents:
                        continue
                    bmagn_value = bmagn['parameter']
                    bmagns.remove(bmagn)
                    break

                # Index calculation depends on model
                if   phase_model == 'RKMPM':
                    # Get indices of participating constituents in phase (order of printed endmembers)
                    indices = []
                    for species in constituent_set:
                        for element in species.constituents:
                            name = element.capitalize()
                            try:
                                indices.append(1 + endmember_names.index(name))
                            except ValueError:
                                print(f'Can\'t find endmember {name}')
                elif phase_model == 'SUBLM':
                    # Get indices of participating constituents in phase (order of printed endmembers)
                    indices = []
                    for sublattice in tc_constituents:
                        for species in sublattice:
                            try:
                                indices.append(1 + flat_constituents.index(species.name))
                            except ValueError:
                                print(f'Can\'t find constituent {constituent}')
                # Now write excess magnetic terms for current constituent_set
                output += f'{len(indices):4}\n'
                # TODO: Get order properly if possible
                order = 1
                output += f'{"".join([f"{ind:4}" for ind in indices])}{order:4}\n'
                # Write excess magnetic parameters line
                output += f' {format_coefficient_mag(tc_value)}{format_coefficient_mag(bmagn_value)}\n'
            # Write end-of-magnetic-excess '0'
            output += f'   0\n'

        # Write excess mixing data
        if   phase_model == 'QKTO':
            detect_query = (
                (where("phase_name") == phase_name) & \
                (where("parameter_type") == "QKT")
            )
            excess_params = list(dbf._parameters.search(detect_query))

            for param in excess_params:
                # Constituents participating in this mixing term
                param_constituents = param['constituent_array'][0]
                n_param_constituents = len(param_constituents)
                output += f'{n_param_constituents:4}\n'

                # Get indices of participating constituents in phase (order of printed endmembers)
                indices = []
                for species in param_constituents:
                    for element in species.constituents:
                        name = element.capitalize()
                        indices.append(1 + endmember_names.index(name))
                output += f'{"".join([f"{index:4}" for index in indices])}'

                # Add 1 to stored exponents to match DAT format
                exponents = [1 + exp for exp in param['exponents']]
                output += f'{"".join([f"{exp:4}" for exp in exponents])}'

                # Parse T coefficients
                equation = param['parameter'].as_coefficients_dict()
                coefficients, extra_parameters, has_extra_parameters = parse_gibbs_coefficients(equation)
                coefficients_string = ''.join(coefficients)
                output += f' {coefficients_string}\n'

            # Write end-of-excess '0'
            output += f'   0\n'
        elif phase_model in ('RKMP','RKMPM'):
            # Get excess mixing parameters
            detect_query = (
                (where("phase_name") == phase_name) & \
                (where("parameter_type") == "L")
            )
            excess_params = list(dbf._parameters.search(detect_query))

            # For RKMP we have to collect all terms for each set of constituents
            unique_constituent_sets = set([param['constituent_array'][0] for param in excess_params])
            for constituent_set in unique_constituent_sets:
                # Get indices of participating constituents in phase (order of printed endmembers)
                indices = []
                for species in constituent_set:
                    for element in species.constituents:
                        name = element.capitalize()
                        try:
                            indices.append(1 + endmember_names.index(name))
                        except ValueError:
                            print(f'Can\'t find endmember {name}')
                # Store all exponents (orders) for constituents
                orders = []
                # Sum coefficients that are for the same order (abnormal case of repeated order)
                equations = []
                for param in excess_params:
                    # Constituents participating in this mixing term
                    param_constituents = param['constituent_array'][0]
                    # Check if param belongs to current constituent_set
                    if param_constituents != constituent_set:
                        continue
                    order = param['parameter_order'] + 1
                    equation = param['parameter']
                    if order in orders:
                        equations[orders.index(order)] += equation
                    else:
                        orders.append(order)
                        equations.append(equation)

                # Now write excess mixing terms for current constituent_set
                output += f'{len(indices):4}\n'
                output += f'{"".join([f"{ind:4}" for ind in indices])}{max(orders):4}\n'
                for order in range(1,max(orders)+1):
                    if order in orders:
                        order_index = orders.index(order)
                        coefficients, extra_parameters, has_extra_parameters = parse_gibbs_coefficients(equations[order_index].as_coefficients_dict())
                        coefficients_string = ''.join(coefficients)
                        output += f' {coefficients_string}\n'
            # Write end-of-excess '0'
            output += f'   0\n'
        elif phase_model in ('SUBL','SUBLM'):
            # Get excess mixing parameters
            detect_query = (
                (where("phase_name") == phase_name) & \
                (where("parameter_type") == "L")
            )
            excess_params = list(dbf._parameters.search(detect_query))

            # For SUBL we have to collect all terms for each set of constituents
            unique_constituent_sets = set([param['constituent_array'] for param in excess_params])
            for constituent_set in unique_constituent_sets:
                # Get indices of participating constituents in phase (order of printed endmembers)
                indices = []
                for sublattice in constituent_set:
                    for species in sublattice:
                        try:
                            indices.append(1 + flat_constituents.index(species.name))
                        except ValueError:
                            print(f'Can\'t find constituent {constituent}')
                # Store all exponents (orders) for constituents
                orders = []
                # Sum coefficients that are for the same order (abnormal case of repeated order)
                equations = []
                for param in excess_params:
                    # Constituents participating in this mixing term
                    param_constituents = param['constituent_array']
                    # Check if param belongs to current constituent_set
                    if param_constituents != constituent_set:
                        continue
                    order = param['parameter_order'] + 1
                    equation = param['parameter']
                    if order in orders:
                        equations[orders.index(order)] += equation
                    else:
                        orders.append(order)
                        equations.append(equation)

                # Now write excess mixing terms for current constituent_set
                output += f'{len(indices):4}\n'
                output += f'{"".join([f"{ind:4}" for ind in indices])}{max(orders):4}\n'
                for order in range(1,max(orders)+1):
                    if order in orders:
                        order_index = orders.index(order)
                        coefficients, extra_parameters, has_extra_parameters = parse_gibbs_coefficients(equations[order_index].as_coefficients_dict())
                        coefficients_string = ''.join(coefficients)
                        output += f' {coefficients_string}\n'
            # Write end-of-excess '0'
            output += f'   0\n'
        elif phase_model in ('SUBG', 'SUBQ'):
            # Get excess mixing parameters
            detect_query = (
                (where("phase_name") == phase_name) & \
                (where("parameter_type") == "MQMX")
            )
            excess_params = list(dbf._parameters.search(detect_query))
            for param in excess_params:
                con = []
                for ion in param['constituent_array']:
                    for species in ion:
                        con.append(flat_constituents.index(species.name) + 1)
                con_string = ''.join([f"{c:4}" for c in con])

                # Determine mixing order by checking number of unique constituents
                if (con[0] == con[1]) or (con[2] == con[3]):
                    mix_order = 3
                else:
                    mix_order = 4
                # Write mixing order
                output += f'{mix_order:4}\n'

                # Write type, constituents, exponents
                exp_string = ''.join([f"{z:4}" for z in param['exponents']])
                output += f' {param["mixing_code"]}{con_string}{exp_string}\n'

                # Write lines of apparent nonsense
                output += '     0.00000000   1.00     0.00000000   1.00     0.00000000   1.00\n'
                output += '     0.00000000   0.00     0.00000000   0.00     0.00000000   0.00\n'

                # Get extra constituent data
                additional_index = 0
                if param["additional_mixing_constituent"].name:
                    additional_index = flat_constituents.index(param["additional_mixing_constituent"].name) + 1

                # Get mixing coefficients
                coefficients, extra_parameters, has_extra_parameters = parse_gibbs_coefficients(param['parameter'].as_coefficients_dict())
                coefficients_string = ''.join(coefficients)

                # Write extra constituent data and mixing coefficients
                output += f'{additional_index:4}{param["additional_mixing_exponent"]:4} {coefficients_string}\n'


            # Write end-of-excess '0'
            output += f'   0\n'

    # Loop over stoichiometric
    for phase_name in stoichiometric_phases:
        # TODO: detect dummies and format accordingly
        # Write phase name
        output += f' {phase_name}\n'

        # Get Gibbs energy parameters
        detect_query = (
            (where("phase_name") == phase_name) & \
            (where("parameter_type") == "G")
        )
        stoichiometric_phase_params = list(dbf._parameters.search(detect_query))

        # Calculate stoichiometry of phase
        endmember = stoichiometric_phase_params[0]
        stoichiometry = [0 for _ in elements_ordered]
        species_index = 0
        for speciesList in endmember['constituent_array']:
            for species in speciesList:
                for element in species.constituents:
                    # Get stoichiometric coefficient of sublattice
                    sublattice_coefficient = dbf.phases[phase_name].sublattices[species_index]
                    stoichiometry[elements_ordered.index(element)] += species.constituents[element]*sublattice_coefficient
                    species_index += 1

        # Writeable stoichiometry
        stoichiometry_string = ''.join([f'{stoich:7.1f}' for stoich in stoichiometry])

        # Determine equation type and number of intervals
        gibbs_equation = endmember['parameter'].args
        eq_type, number_of_intervals, gibbs_parameters = parse_gibbs_coefficients_piecewise(gibbs_equation)

        # Write equation type and stoichiometry line
        output += f'{eq_type:4} {number_of_intervals:2}{stoichiometry_string}\n'
        # Write Gibbs parameters line
        output += gibbs_parameters





    fd.write(reflow_text(output, linewidth=maxlen))

def parse_gibbs_coefficients_piecewise(piecewise_equation):
    # Set eq_type to 1 by default
    # TODO: detect magnatic parameters and set eq_type
    eq_type = 1
    # Pattern is (equation, temperature interval)*n_intervals, then two extra parameters
    number_of_intervals = int((len(piecewise_equation) - 2) / 2)
    # If one interval has extra parameters, must use equation type that supports them
    has_extra_parameters = False
    gibbs_parameters = ''
    for interval in range(number_of_intervals):
        equation = piecewise_equation[interval*2].as_coefficients_dict()
        # Parse coefficients from equation
        coefficients, extra_parameters, interval_has_extra_parameters = parse_gibbs_coefficients(equation)
        has_extra_parameters = has_extra_parameters or interval_has_extra_parameters

        coefficients_string = '     '.join(coefficients)
        # Get temperature range part of equation
        temperature_range = piecewise_equation[interval*2 + 1]

        # This is rough, not sure how to reliably extract bounds from pairs of inequalities
        # This method is certain to break if order ever varies
        try:
            max_t = float(temperature_range.args[0].args[1])
        except RuntimeError:
            max_t = float(temperature_range.args[1].args[1])
        # Trailing 0 padding for temperatures is weird
        max_t_string = f'{max_t:.3f}'.ljust(9,'0')

        # Put the line together
        gibbs_parameters += f'  {max_t_string}     {"".join(coefficients)}\n'

        # Add extra parameters if necessary
        if has_extra_parameters:
            # Base parameter set to 4 if there are extra parameters
            eq_type = 4
            if extra_parameters:
                extra_parameters_string = ''
                for parameter in extra_parameters:
                    extra_parameters_string += f' {parameter[0]} {parameter[1]}'
                gibbs_parameters += f' {len(extra_parameters)}{extra_parameters_string}\n'
            else:
                gibbs_parameters += f' 1 0.00000000       0.00\n'

    return eq_type, number_of_intervals, gibbs_parameters

def parse_gibbs_coefficients(equation):
    # Initialize all standard coefficients to 0
    coefficients = ['0.00000000     ' for _ in range(6)]
    # Arrays for extra parameters
    extra_parameters = []
    has_extra_parameters = False
    # Check order in temperature and put coefficient in matching slot
    for t_order in equation:
        # Truncation error is introduced here
        coeff = float(equation[t_order])
        coeff_string = format_coefficient(coeff)

        # Compare order to standard orders for coefficients
        if   str(t_order) == '1':
            coefficients[0] = coeff_string
        elif str(t_order) == 'T':
            coefficients[1] = coeff_string
        elif str(t_order) == 'T*log(T)':
            coefficients[2] = coeff_string
        elif str(t_order) == 'T**2.0':
            coefficients[3] = coeff_string
        elif str(t_order) == 'T**3.0':
            coefficients[4] = coeff_string
        elif str(t_order) == 'T**(-1.0)':
            coefficients[5] = coeff_string
        # These are common extra parameters
        elif str(t_order) == 'log(T)':
            has_extra_parameters = True
            extra_parameters.append((coeff_string,'99.00'))
        elif str(t_order) == 'T**0.5':
            has_extra_parameters = True
            extra_parameters.append((coeff_string,' 0.50'))
        # TODO: Add other supported extra parameters (i.e. T**X)
        # TODO: Handle non-supported parameters properly
        elif coeff == 1:
            # This indicates the constant term had order/coefficient flipped
            coeff_string = format_coefficient(float(t_order))
            coefficients[0] = coeff_string
        else:
            # Try parsing as T**x polynomial
            splitOrder = str(t_order).split('**')
            if len(splitOrder) == 2:
                if splitOrder[0] == 'T':
                    try:
                        orderStrip = splitOrder[1].replace('(','').replace(')','')
                        extra_parameters.append((coeff_string,f'{float(orderStrip):.2f}'))
                        has_extra_parameters = True
                        continue
                    except ValueError:
                        pass
            print(f'WARNING: Skipped parameter with order {t_order} and coefficient {coeff_string}')

    return coefficients, extra_parameters, has_extra_parameters

def format_coefficient(coeff):
    # The formatting is inconsistent, so unfortunately each value range is custom
    if   coeff == 0:
        coeff_string = '0.00000000     '
    elif abs(coeff) < 0.1 or abs(coeff) >= 1e8:
        coeff_string = f'{coeff*10: .7E}'[:15]
        if coeff < 0:
            coeff_string = '-.' + coeff_string[1] + coeff_string[3:] + ' '
        else:
            coeff_string = '0.' + coeff_string[1] + coeff_string[3:] + ' '
    elif abs(coeff) < 1:
        coeff_string = f'{coeff: .16f}'[:10] + '     '
        if coeff < 0:
            coeff_string = ' -' + coeff_string[2:]
    else:
        coeff_string = f'{coeff: .16f}'[:10] + '     '

    return coeff_string

def format_coefficient_mag(coeff):
    # The formatting is inconsistent, so unfortunately each value range is custom
    if   coeff == 0:
        coeff_string = '0.000000     '
    elif abs(coeff) < 0.1 or abs(coeff) >= 1e6:
        coeff_string = f'{coeff*10: .5E}'[:13]
        if coeff < 0:
            coeff_string = '-.' + coeff_string[1] + coeff_string[3:] + ' '
        else:
            coeff_string = '0.' + coeff_string[1] + coeff_string[3:] + ' '
    elif abs(coeff) < 1:
        coeff_string = f'{coeff: .16f}'[:8] + '     '
        if coeff < 0:
            coeff_string = ' -' + coeff_string[2:]
    else:
        coeff_string = f'{coeff: .16f}'[:8] + '     '

    return coeff_string

def make_constituent_mapping(constituents, endmember_params):
    # Match endmembers to constituents they are composed of
    constituent_mapping = [[] for _ in range(len(constituents))]
    for endmember in endmember_params:
        sublattice = 0
        for speciesList in endmember['constituent_array']:
            for species in speciesList:
                constituent_mapping[sublattice].append(constituents[sublattice].index(species.name) + 1)
                sublattice += 1
    return constituent_mapping

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

Database.register_format("dat", read=read_cs_dat, write=write_cs_dat)
