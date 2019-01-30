import numpy as np


class BinaryCompSet():
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
        return "<BinaryCompSet{0}(T={1:0.3f}, X({2})={3:0.3f})>".format(self.phase_name, self.temperature, self.indep_comp, self.composition)

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


    def xdiscrepancy(self, other, ignore_phase=False):
        """
        Calculate the composition discrepancy (absolute difference) between this
        composition set and another.

        Parameters
        ----------
        other : BinaryCompSet
        ignore_phase : bool
            If False, unlike phases will give infinite discrepancy. If True, we
            only care about the composition and the real discrepancy will be returned.

        Returns
        -------
        np.float64

        """
        if not ignore_phase and self.phase_name != other.phase_name:
            return np.infty
        else:
            return np.abs(self.composition - other.composition)

    def ydiscrepancy(self, other):
        """
        Calculate the discrepancy (absolute differences) between the site
        fractions of this composition set and another as an array of discrepancies.

        Parameters
        ----------
        other : BinaryCompSet

        Returns
        -------
        Array of np.float64

        Notes
        -----
        The phases must match for this to be meaningful.

        """
        if self.phase_name != other.phase_name:
            return np.infty
        else:
            return np.abs(self.site_fracs - other.site_fracs)

    def ydiscrepancy_max(self, other):
        """
        Calculate the maximum discrepancy (absolute difference) between the site
        fractions of this composition set and another.

        Parameters
        ----------
        other : BinaryCompSet

        Returns
        -------
        np.float64

        Notes
        -----
        The phases must match for this to be meaningful.

        """
        if self.phase_name != other.phase_name:
            return np.infty
        else:
            return np.max(np.abs(self.site_fracs - other.site_fracs))


    def Tdiscrepancy(self, other, ignore_phase=False):
        """
        Calculate the temperature discrepancy (absolute difference) between this
        composition set and another.

        Parameters
        ----------
        other : BinaryCompSet
        ignore_phase : bool
            If False, unlike phases will give infinite discrepancy. If True, we
            only care about the composition and the real discrepancy will be returned.

        Returns
        -------
        np.float64

        """
        if not ignore_phase and self.phase_name != other.phase_name:
            return np.infty
        else:
            return np.abs(self.temperature - other.temperature)


    @staticmethod
    def mean_composition(compsets):
        """
        Return the mean composition of a list of composition sets.

        Parameters
        ----------
        compsets : list of composition sets

        Returns
        -------
        np.float

        """
        return np.mean([c.composition for c in compsets])


    @staticmethod
    def composition_sorted(compsets):
        """
        Sort the BinaryCompSets by increasing composition

        Parameters
        ----------
        compsets : list
            List of BinaryCompSet objects

        Returns
        -------
        list
        """
        _composition_key_func = lambda c: c.composition
        return sorted(compsets, key=_composition_key_func)
