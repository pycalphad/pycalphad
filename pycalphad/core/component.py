class Component(object):
    # name, mass, charge
    def __new__(cls, obj):
        if isinstance(obj, Component):
            return obj
        retobj = super(Component, cls).__new__(cls, obj)
        # Species
        # string
        return retobj