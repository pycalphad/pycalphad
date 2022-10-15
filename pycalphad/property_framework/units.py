import pint
import numpy as np

ureg = pint.UnitRegistry()
ureg.define('atom = 1/avogadro_number * mol')
Q_ = ureg.Quantity

energy_base_units = GM_base_units = 'J / mol'
energy_display_units = GM_display_units = 'kJ / mol'
energy_display_name = GM_display_name = 'Gibbs Energy'
enthalpy_base_units = HM_base_units = GM_base_units
enthalpy_display_units = HM_display_units = GM_display_units
enthalpy_display_name = HM_display_name = 'Enthalpy'
entropy_base_units = SM_base_units = 'J / mol / K'
entropy_display_units = SM_display_units = 'J / mol / K'
entropy_display_name = SM_display_name = 'Entropy'

# g/mol
# It would be 'better' to pull this information from databases, but it would require significant design changes
_molmass = \
    {"H": 1.008, "HE": 4.003, "LI": 6.941, "BE": 9.012, "B": 10.811, "C": 12.011, "N": 14.007, "O": 15.999,
     "F": 18.998, "NE": 20.18, "NA": 22.99, "MG": 24.305, "AL": 26.982, "SI": 28.086, "P": 30.974, "S": 32.065,
     "CL": 35.453, "AR": 39.948, "K": 39.098, "CA": 40.078, "SC": 44.956, "TI": 47.867, "V": 50.942, "CR": 51.996,
     "MN": 54.938, "FE": 55.845, "CO": 58.933, "NI": 58.693, "CU": 63.546, "ZN": 65.39, "GA": 69.723, "GE": 72.64,
     "AS": 74.922, "SE": 78.96, "BR": 79.904, "KR": 83.8, "RB": 85.468, "SR": 87.62, "Y": 88.906, "ZR": 91.224,
     "NB": 92.906, "MO": 95.94, "TC": 98, "RU": 101.07, "RH": 102.906, "PD": 106.42, "AG": 107.868, "CD": 112.411,
     "IN": 114.818, "SN": 118.71, "SB": 121.76, "TE": 127.6, "I": 126.905, "XE": 131.293, "CS": 132.906,
     "BA": 137.327, "LA": 138.906, "CE": 140.116, "PR": 140.908, "ND": 144.24, "PM": 145, "SM": 150.36,
     "EU": 151.964, "GD": 157.25, "TB": 158.925, "DY": 162.5, "HO": 164.93, "ER": 167.259, "TM": 168.934,
     "YB": 173.04, "LU": 174.967, "HF": 178.49, "TA": 180.948, "W": 183.84, "RE": 186.207, "OS": 190.23,
     "IR": 192.217, "PT": 195.078, "AU": 196.967, "HG": 200.59, "TL": 204.383, "PB": 207.2, "BI": 208.98,
     "PO": 209, "AT": 210, "RN": 222, "FR": 223, "RA": 226, "AC": 227, "TH": 232.038, "PA": 231.036, "U": 238.029,
     "NP": 237, "PU": 244, "AM": 243, "CM": 247, "BK": 247, "CF": 251, "ES": 252, "FM": 257, "MD": 258, "NO": 259,
     "LR": 262, "RF": 261, "DB": 262, "SG": 266, "BH": 264, "HS": 277, "MT": 268
     }

def _conversions_per_formula_unit(compset):
    components = compset.phase_record.nonvacant_elements
    num_components = len(components)
    moles_per_fu = np.zeros((num_components,1))
    for comp_idx in range(num_components):
        compset.phase_record.formulamole_obj(moles_per_fu[comp_idx, :], compset.dof, comp_idx)
    # now we have 'moles per formula unit'
    # need to convert by adding molecular weight of each element
    grams_per_mol = np.array(list(_molmass[el] for el in components), dtype='float')
    grams_per_fu = np.dot(grams_per_mol, moles_per_fu)
    return moles_per_fu.sum(), grams_per_fu

def unit_conversion_context(compsets, prop):
    context = pint.Context()
    # these will be something/mol by convention
    # XXX: This is a very rough check
    if not ('/ mol' in str(prop.base_units)):
        return context
    base_units = (ureg.Unit(prop.base_units) * ureg.Unit('mol'))
    molar_weight = 0.0 # g/mol-atom
    for compset in compsets:
        if compset.NP > 0:
            moles_per_fu, grams_per_fu = _conversions_per_formula_unit(compset)
            grams_per_mol_atoms = (compset.NP / moles_per_fu) * grams_per_fu
            molar_weight += grams_per_mol_atoms
    molar_weight = Q_(molar_weight, 'g/mol')
    per_moles = ureg.get_dimensionality(ureg.Unit('{} / mol'.format(base_units)))
    per_mass = ureg.get_dimensionality(ureg.Unit('{} / g'.format(base_units)))

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