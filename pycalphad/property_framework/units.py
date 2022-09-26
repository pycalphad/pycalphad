import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

energy_base_units = GM_base_units = 'J / mol'
energy_display_units = GM_display_units = 'kJ / mol'
energy_display_name = GM_display_name = 'Gibbs Energy'
enthalpy_base_units = HM_base_units = GM_base_units
enthalpy_display_units = HM_display_units = GM_display_units
enthalpy_display_name = HM_display_name = 'Enthalpy'