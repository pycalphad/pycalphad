import pint
import numpy as np
import numpy.typing as npt
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pycalphad.property_framework import ComputableProperty

ureg = pint.UnitRegistry(preprocessors=[lambda s: s.replace('%', ' percent ')])
ureg.define('atom = 1/avogadro_number * mol')
ureg.define('fraction = []')
ureg.define('percent = 1e-2 fraction = %')
ureg.define('ppm = 1e-6 fraction')
pint.set_application_registry(ureg)
Q_ = ureg.Quantity
DimensionalityError = pint.DimensionalityError

def as_quantity(prop: "ComputableProperty", qt: npt.ArrayLike):
    if not isinstance(qt, Q_):
        return Q_(qt, prop.display_units)
    else:
        return qt

energy_implementation_units = GM_implementation_units = 'J / mol'
energy_display_units = GM_display_units = 'J / mol'
energy_display_name = GM_display_name = 'Gibbs Energy'
G_implementation_units = 'J'
G_display_units = 'J'
G_display_name = 'Gibbs Energy'
enthalpy_implementation_units = HM_implementation_units = GM_implementation_units
enthalpy_display_units = HM_display_units = GM_display_units
enthalpy_display_name = HM_display_name = 'Enthalpy'
H_implementation_units = 'J'
H_display_units = 'J'
H_display_name = 'Enthalpy'
entropy_implementation_units = SM_implementation_units = 'J / mol / K'
entropy_display_units = SM_display_units = 'J / mol / K'
entropy_display_name = SM_display_name = 'Entropy'

def _conversions_per_formula_unit(compset):
    components = compset.phase_record.nonvacant_elements
    num_components = len(components)
    moles_per_fu = np.zeros((num_components,1))
    for comp_idx in range(num_components):
        compset.phase_record.formulamole_obj(moles_per_fu[comp_idx, :], compset.dof, comp_idx)
    # now we have 'moles per formula unit'
    # need to convert by adding molecular weight of each element
    grams_per_mol = np.array(compset.phase_record.molar_masses, dtype='float')
    grams_per_fu = np.dot(grams_per_mol, moles_per_fu)
    return moles_per_fu.sum(), grams_per_fu

def unit_conversion_context(compsets, prop):
    context = pint.Context()
    # these will be something/mol by convention
    # XXX: This is a very rough check
    if not ('/ mol' in str(prop.implementation_units)):
        return context
    implementation_units = (ureg.Unit(prop.implementation_units) * ureg.Unit('mol'))
    molar_weight = 0.0 # g/mol-atom
    for compset in compsets:
        if compset.NP > 0:
            moles_per_fu, grams_per_fu = _conversions_per_formula_unit(compset)
            grams_per_mol_atoms = (compset.NP / moles_per_fu) * grams_per_fu
            molar_weight += grams_per_mol_atoms
    molar_weight = Q_(molar_weight, 'g/mol')
    per_moles = ureg.get_dimensionality(ureg.Unit('{} / mol'.format(implementation_units)))
    per_mass = ureg.get_dimensionality(ureg.Unit('{} / g'.format(implementation_units)))

    context.add_transformation(
        per_moles,
        per_mass,
        lambda ureg, x: (x / molar_weight).to_reduced_units()
    )
    context.add_transformation(
        per_mass,
        per_moles,
        lambda ureg, x: (x * molar_weight).to_reduced_units()
    )

    return context