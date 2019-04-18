import numpy as np
import warnings

class CompSet():
    """
    Composition set for 2D representations of binary, ternary or multicomponent equilibrium.
    """
    # tolerances for defining equality
    SITE_FRAC_ATOL = 0.001
    TEMPERATURE_ATOL = 0.1

    def __init__(self, phase_name, temperature, indep_comp, composition, site_fracs):
        self.phase_name = phase_name
        self.temperature = temperature
        self.indep_comp = indep_comp
        self.composition = composition
        self.site_fracs = site_fracs

    def __repr__(self,):
        return "CompSet({0}:<T={1:0.3f}, X({2})={3:0.3f}>)".format(self.phase_name, self.temperature, self.indep_comp, self.composition)

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

    def isclose(self, other, comp_tol=0.01, temp_tol=1):
        if self.phase_name == other.phase_name:
            if self.xdiscrepancy(other) < comp_tol and self.Tdiscrepancy(other) < temp_tol:
                return True
        return False

    @classmethod
    def from_dataset_vertices(cls, ds, indep_comp, indep_comp_idx, num_vertex):
        compsets = []
        Phase = ds.Phase.values.squeeze()
        T = ds.T.values.squeeze()
        X = ds.X.values.squeeze()
        Y = ds.Y.values.squeeze()
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


class CompSet2D():
    """
    Pair of composition sets

    Attributes
    ----------
    compsets : list of CompSet
        CompSets sorted by composition
    a : CompSet
        Composition set in the pair with the lower composition
    b : CompSet
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
        other : CompSet2D

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
        other : CompSet2D

        Returns
        -------
        float
        """
        if self.phases == other.phases:
            return np.abs(self.compositions - other.compositions)
        else:
            return np.full(self.compositions.shape, np.infty)
