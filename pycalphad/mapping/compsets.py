class BinaryCompSet():
    def __init__(self, phase_name, temperature, indep_comp, composition, site_fracs):
        self.phase_name = phase_name
        self.temperature = temperature
        self.indep_comp = indep_comp
        self.composition = composition
        self.site_fracs = site_fracs

    def __repr__(self,):
        return "BinaryCompSet<{0}(T={1}, X({2})={3})>".format(self.phase_name, self.temperature, self.indep_comp, self.composition)

    def __str__(self,):
        return self.__repr__()

    @classmethod
    def from_dataset_vertex(cls, ds):
        def get_val(da):
            return da.values.flatten()[0]
        def get_vals(da):
            return da.values.flatten()
        indep_comp = [c for c in ds.coords if 'X_' in c][0][2:]

        return BinaryCompSet(get_val(ds.Phase),
                             get_val(ds.T),
                             indep_comp,
                             get_val(ds.X.sel(component=indep_comp)),
                             get_vals(ds.Y)
                            )
