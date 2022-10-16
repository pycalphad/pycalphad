from pycalphad.codegen.phase_record_factory import PhaseRecordFactory


def build_phase_records(dbf, comps, phases, state_variables, models, output='GM',
                        callables=None, parameters=None, verbose=False,
                        build_gradients=True, build_hessians=True
                        ):
    if output != 'GM':
        raise ValueError('build_phase_records is deprecated and no longer works when the output keyword '
                         'is changed from the default. Remove the keyword, and then use the PhaseRecord.prop_* API '
                         'in downstream functions instead.')
    return PhaseRecordFactory(dbf, comps, state_variables, models, parameters=parameters)
