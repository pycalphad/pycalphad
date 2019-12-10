import numpy as np
import warnings

class BinaryCompset():
    """
    Composition set for representations of a phase equilibria in a two
    component system with a temperature condition.

    Parameters
    ----------
    phase_name : str
        Name of phase
    temperature : float
        Temperature corresponding the the calculation
    indep_comp : str
        Name of the independent component
    composition : float
        Mole fraction of the independent component
    site_fracs : np.ndarray
        Array of floats corresponding to the site fractions.

    Notes
    -----
    In the future, this representation could be phased out if `equilibrium` returned composition sets.
    """
    # tolerances for defining equality between composition sets
    SITE_FRAC_ATOL = 0.001
    TEMPERATURE_ATOL = 0.1

    def __init__(self, phase_name, temperature, indep_comp, composition, site_fracs):
        self.phase_name = phase_name
        self.temperature = temperature
        self.indep_comp = indep_comp
        self.composition = composition
        self.site_fracs = site_fracs

    def __repr__(self,):
        return "<BinaryCompset({0}, T={1:0.3f}, X({2})={3:0.3f})>".format(self.phase_name, self.temperature, self.indep_comp, self.composition)

    def __str__(self,):
        return self.__repr__()

    def __eq__(self, other):
        if hasattr(other, 'phase_name') and hasattr(other, 'site_fracs') and hasattr(other, 'temperature'):
            same_phase = self.phase_name == other.phase_name
            site_frac_close = np.all(np.isclose(self.site_fracs, other.site_fracs, atol=self.__class__.SITE_FRAC_ATOL))
            temp_close = np.isclose(self.temperature, other.temperature, atol=self.__class__.TEMPERATURE_ATOL)
            return same_phase and site_frac_close and temp_close
        else:
            return False

    @classmethod
    def from_dataset_vertices(cls, ds, indep_comp, indep_comp_idx, num_vertex):
        compsets = []
        Phase = ds["Phase"].squeeze()
        T = np.array(ds["T"]).squeeze()
        X = ds["X"].squeeze()
        Y = ds["Y"].squeeze()
        for i in range(num_vertex):
            phase_name = Phase[i]
            if phase_name != '':
                compsets.append(cls(str(phase_name),
                                    T,
                                    indep_comp,
                                    X[i, indep_comp_idx],
                                    Y[i, :],
                                    ))
        return compsets


class CompsetPair():
    """
    Pair of binary composition sets that make up an equilibrium

    Parameters
    ----------
    compsets : list of BinaryCompset

    Attributes
    ----------
    compsets : list of BinaryCompset
        CompSets sorted by composition
    a : BinaryCompset
        Composition set in the pair with the lower composition
    b : BinaryCompset
        Composition set in the pair with the higher composition
    phases : list of str
        List of phase names, sorted by composition
    unique_phases : set of str
        Set of phase names
    compositions : np.ndarray
        Array of compositions, sorted
    temperature : float
        Temperature of the 2D compsets
    same_phase : bool
        Whether the composition sets are for the same phase
    mean_composition : float
        Mean composition of the CompSets
    max_composition : float
        Max composition of the CompSets
    min_composition : float
        Min composition of the CompSets
    """
    def __init__(self, compsets):
        _composition_key_func = lambda c: c.composition
        sorted_compsets = sorted(compsets, key=_composition_key_func)
        phases = [c.phase_name for c in sorted_compsets]
        compositions = np.array([c.composition for c in sorted_compsets])
        a = sorted_compsets[0]
        b = sorted_compsets[1]

        self._orig_compsets = compsets  # with original sorting
        self.compsets = sorted_compsets
        self.phases = phases
        self.unique_phases = frozenset(phases)
        self.a = a
        self.b = b
        self.same_phase = a.phase_name == b.phase_name
        self.compositions = compositions
        self.mean_composition = np.mean(compositions)
        self.min_composition = a.composition
        self.max_composition = b.composition
        if a.temperature == b.temperature:
            self.temperature = a.temperature
        else:
            warnings.warn("Temperatures are different for CompSet objects {}. Assuming that the pair temperature is ".format(sorted_compsets, a.temperature))
            self.temperature = a.temperature

    def xdiscrepancy(self, ignore_phase=False):
        """
        Calculate the composition discrepancy (absolute difference) between this
        composition set and another.

        Parameters
        ----------
        ignore_phase : bool
            If False, unlike phases will give infinite discrepancy. If True, we
            only care about the composition and the real discrepancy will be returned.

        Returns
        -------
        np.float64

        """
        if self.same_phase or ignore_phase:
            return np.abs(self.a.composition - self.b.composition)
        else:
            return np.infty

    def ydiscrepancy(self):
        """
        Calculate the discrepancy (absolute differences) between the site
        fractions of the composition sets as an array of discrepancies.

        Returns
        -------
        Array of np.float64

        Notes
        -----
        The phases must match for this to be meaningful.

        """
        if self.same_phase:
            return np.abs(self.a.site_fracs - self.b.site_fracs)
        else:
            return np.infty

    def ydiscrepancy_max(self):
        """
        Calculate the maximum discrepancy (absolute difference) between the site
        fractions of the composition sets.

        Returns
        -------
        np.float64

        Notes
        -----
        The phases must match for this to be meaningful.

        """
        return np.max(np.abs(self.ydiscrepancy()))

    def Tdiscrepancy(self, other):
        """
        Calculate the temperature discrepancy (absolute difference) between this
        pair of composition sets and another.

        Parameters
        ----------
        other : CompsetPair

        Returns
        -------
        np.float64

        """
        return np.abs(self.temperature - other.temperature)

    def __repr__(self,):
        compset_strs = ", ".join(["{0}:<X({1})={2:0.3f}>".format(c.phase_name, c.indep_comp, c.composition) for c in self.compsets])
        return "CompSet2D(T={0:0.3f}: ({1})".format(self.temperature, compset_strs)

    def __str__(self,):
        return self.__repr__()

    def pairwise_xdiscrepancy(self, other):
        """
        Compute the ordered composition discrepancy between this and another
        pair of CompSet2D objects.

        Parameters
        ----------
        other : CompsetPair

        Returns
        -------
        float
        """
        if self.phases == other.phases:
            return np.abs(self.compositions - other.compositions)
        else:
            return np.full(self.compositions.shape, np.infty)


def get_compsets(eq_dataset, indep_comp=None, indep_comp_index=None):
    """
    Return a CompSet2D object if a pair of composition sets is found in an
    equilibrium dataset. Otherwise return None.

    Parameters
    ----------
    eq_dataset :
    indep_comp :
    indep_comp_index :

    Returns
    -------
    CompsetPair
    """
    if indep_comp is None:
        indep_comp = [c for c in eq_dataset.coords if 'X_' in c][0][2:]
    if indep_comp_index is None:
        indep_comp_index = eq_dataset.component.values.tolist().index(indep_comp)
    extracted_compsets = BinaryCompset.from_dataset_vertices(eq_dataset, indep_comp, indep_comp_index, 3)
    if len(extracted_compsets) == 2:
        return CompsetPair(extracted_compsets)
    else:
        return None


def find_two_phase_region_compsets(hull, temperature, indep_comp, indep_comp_idx, misc_gap_tol=0.1, minimum_composition=None):
    """
    From a 1D convex hull at constant T and P, return the composition sets for
    a two phase region or that have the smallest index composition coordinate

    Parameters
    ----------
    hull : EquilibriumResult
        Equilibrium-like from pycalphad that has a `Phase` Data variable.
    temperature : float
        Temperature that the calculation was performed at
    indep_comp : str
        Name of the independent component
    indep_comp_idx : str
        Index of the independent component in the the sorted pure elements
    misc_gap_tol : float
        If any site fractions are different by at least this amount, the
        composition sets are considered distinct and in a miscibility gap.
    minimum_composition : float
        Minimum composition in the convex hull to search for composition sets

    Returns
    -------
    CompsetPair

    """
    phases = hull.Phase.squeeze()
    compositions = hull.X.squeeze()
    site_fracs = hull.Y.squeeze()
    grid_shape = phases.shape[:-1]
    num_phases = phases.shape[-1]
    it = np.nditer(np.empty(grid_shape), flags=['multi_index'])  # empty grid for indexing
    while not it.finished:
        idx = it.multi_index
        cs = []
        # TODO: assumption of only two phases, seems like the third phase index can have bad points
        # Three phases is probably an error anyways...
        if minimum_composition is not None and np.all(compositions[idx][:, indep_comp_idx][:2] < minimum_composition):
            it.iternext()
            continue
        for i in np.arange(num_phases):
            if str(phases[idx][i]) != '':
                stable_composition_sets = BinaryCompset(str(phases[idx][i]), temperature, indep_comp, compositions[idx][i, indep_comp_idx], site_fracs[idx][i, :])
                cs.append(stable_composition_sets)
        if len(cs) == 2:
            compsets = CompsetPair(cs)
            if len(compsets.unique_phases) == 2:
                return compsets  # found a multiphase region
            else:
                # Same phase, either single phase region or a miscibility gap.
                y_discrep = compsets.ydiscrepancy()
                if np.any(y_discrep[~np.isnan(y_discrep)] > misc_gap_tol):
                    return compsets  # miscibility gap
        it.iternext()
    return None
