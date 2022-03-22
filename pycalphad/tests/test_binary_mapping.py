from pycalphad import variables as v
from pycalphad.plot.binary.compsets import BinaryCompset, CompsetPair
from pycalphad.plot.binary.map import map_binary
from pycalphad.plot.binary.zpf_boundary_sets import TwoPhaseRegion, ZPFBoundarySets
from pycalphad.tests.fixtures import select_database, load_database


@select_database("alfe.tdb")
def test_binary_mapping(load_database):
    """
    Binary mapping should return a ZPFBoundarySets object
    """
    dbf = load_database()
    my_phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'AL5FE2',
                 'AL2FE', 'AL13FE4', 'AL5FE4']
    comps = ['AL', 'FE', 'VA']
    conds = {v.T: (1200, 1300, 50), v.P: 101325, v.X('AL'): (0, 1, 0.2)}
    zpf_boundaries = map_binary(dbf, comps, my_phases, conds)
    num_boundaries = len(zpf_boundaries.all_compsets)
    assert num_boundaries > 0
    # calling binplot again can add more boundaries
    map_binary(dbf, comps, my_phases, conds, boundary_sets=zpf_boundaries)
    assert len(zpf_boundaries.all_compsets) == 2*num_boundaries


def test_two_phase_region_usage():
    """A new pair of compsets at a slightly higher temperature should be in the region and can be added"""
    compsets_298 = CompsetPair([
        BinaryCompset('P1', 298.15, 'B', 0.5, [0.5, 0.5]),
        BinaryCompset('P2', 298.15, 'B', 0.8, [0.2, 0.8]),
    ])

    compsets_300 = CompsetPair([
        BinaryCompset('P1', 300, 'B', 0.5, [0.5, 0.5]),
        BinaryCompset('P2', 300, 'B', 0.8, [0.2, 0.8]),
    ])

    tpr = TwoPhaseRegion(compsets_298)  # Initial compsets for P1 and P2 at 298 K
    assert tpr.compsets_belong_in_region(compsets_300)
    tpr.add_compsets(compsets_300)
    assert len(tpr.compsets) == 2


def test_two_phase_region_outside_temperature_tolerance_does_not_belong():
    """A CompsetPair with very different temperature should not belong in the TwoPhaseRegion"""
    compsets_300 = CompsetPair([
        BinaryCompset('P1', 300, 'B', 0.5, [0.5, 0.5]),
        BinaryCompset('P2', 300, 'B', 0.8, [0.2, 0.8]),
    ])

    compsets_500 = CompsetPair([
        BinaryCompset('P1', 500, 'B', 0.5, [0.5, 0.5]),
        BinaryCompset('P2', 500, 'B', 0.8, [0.2, 0.8]),
    ])

    tpr = TwoPhaseRegion(compsets_300)  # Initial compsets for P1 and P2 at 300 K
    assert tpr.compsets_belong_in_region(compsets_500) is False


def test_two_phase_region_expands_as_compsets_are_added():
    """A CompsetPair with very different temperature should not belong in the TwoPhaseRegion"""
    compsets_300 = CompsetPair([
        BinaryCompset('P1', 300, 'B', 0.5, [0.5, 0.5]),
        BinaryCompset('P2', 300, 'B', 0.8, [0.2, 0.8]),
    ])

    compsets_305 = CompsetPair([
        BinaryCompset('P1', 305, 'B', 0.5, [0.5, 0.5]),
        BinaryCompset('P2', 305, 'B', 0.8, [0.2, 0.8]),
    ])

    compsets_312 = CompsetPair([
        BinaryCompset('P1', 312, 'B', 0.5, [0.5, 0.5]),
        BinaryCompset('P2', 312, 'B', 0.8, [0.2, 0.8]),
    ])

    tpr = TwoPhaseRegion(compsets_300)  # Initial compsets for P1 and P2 at 300 K
    # compsets don't belong because they are outside the temperature tolerance (10 K)
    assert tpr.compsets_belong_in_region(compsets_312) is False
    assert tpr.compsets_belong_in_region(compsets_305)
    tpr.add_compsets(compsets_305)
    # 312 K compsets could be added now that the 305 K is within 10 K.
    assert  tpr.compsets_belong_in_region(compsets_312)


def test_two_phase_region_new_phases_does_not_belong():
    """A new pair of compsets with different phases should not be in the TwoPhaseRegion"""
    compsets_298 = CompsetPair([
        BinaryCompset('P1', 298.15, 'B', 0.5, [0.5, 0.5]),
        BinaryCompset('P2', 298.15, 'B', 0.8, [0.2, 0.8]),
    ])

    compsets_300_diff_phases = CompsetPair([
        BinaryCompset('P2', 300, 'B', 0.5, [0.5, 0.5]),
        BinaryCompset('P3', 300, 'B', 0.8, [0.2, 0.8]),
    ])

    tpr = TwoPhaseRegion(compsets_298)  # Initial compsets for P1 and P2 at 298 K
    assert tpr.compsets_belong_in_region(compsets_300_diff_phases) is False


def test_adding_compsets_to_zpf_boundary_sets():
    """Test that new composition sets can be added to ZPFBoundarySets successfully."""
    compsets_298 = CompsetPair([
        BinaryCompset('P1', 298.15, 'B', 0.5, [0.5, 0.5]),
        BinaryCompset('P2', 298.15, 'B', 0.8, [0.2, 0.8]),
    ])

    compsets_300 = CompsetPair([
        BinaryCompset('P1', 300, 'B', 0.5, [0.5, 0.5]),
        BinaryCompset('P2', 300, 'B', 0.8, [0.2, 0.8]),
    ])

    compsets_300_diff_phases = CompsetPair([
        BinaryCompset('P2', 300, 'B', 0.5, [0.5, 0.5]),
        BinaryCompset('P3', 300, 'B', 0.8, [0.2, 0.8]),
    ])

    zpfbs = ZPFBoundarySets(['A', 'B'], v.X('B'))
    assert zpfbs.components == ['A', 'B']
    assert len(zpfbs.two_phase_regions) == 0
    assert len(zpfbs.all_compsets) == 0

    zpfbs.add_compsets(compsets_298)
    assert len(zpfbs.all_compsets) == 1
    assert len(zpfbs.two_phase_regions) == 1

    zpfbs.add_compsets(compsets_300)  # same region, different temperature
    assert len(zpfbs.all_compsets) == 2
    assert len(zpfbs.two_phase_regions) == 1

    zpfbs.add_compsets(compsets_300_diff_phases)  # new region, different phases
    assert len(zpfbs.all_compsets) == 3
    assert len(zpfbs.two_phase_regions) == 2


def test_rebulding_zpf_boundary_sets_regions():
    """Test that three regions generated by ZPFBoundarySets can correctly be rebuilt to two regions"""

    compsets_298 = CompsetPair([
        BinaryCompset('P1', 298.15, 'B', 0.5, [0.5, 0.5]),
        BinaryCompset('P2', 298.15, 'B', 0.8, [0.2, 0.8]),
    ])

    compsets_310 = CompsetPair([
        BinaryCompset('P1', 310, 'B', 0.5, [0.5, 0.5]),
        BinaryCompset('P2', 310, 'B', 0.8, [0.2, 0.8]),
    ])

    compsets_300_diff_phases = CompsetPair([
        BinaryCompset('P2', 300, 'B', 0.5, [0.5, 0.5]),
        BinaryCompset('P3', 300, 'B', 0.8, [0.2, 0.8]),
    ])

    zpfbs = ZPFBoundarySets(['A', 'B'], v.X('B'))

    # Initial compsets
    zpfbs.add_compsets(compsets_298)
    assert len(zpfbs.all_compsets) == 1
    assert len(zpfbs.two_phase_regions) == 1

    # Compsets added create a new region because phases changed
    zpfbs.add_compsets(compsets_300_diff_phases)
    assert len(zpfbs.all_compsets) == 2
    assert len(zpfbs.two_phase_regions) == 2

    # Compsets added create a new region because phases the temperature is out of tolerance
    zpfbs.add_compsets(compsets_310)
    assert len(zpfbs.all_compsets) == 3
    assert len(zpfbs.two_phase_regions) == 3

    # Rebuild the regions with a larger tolerance should create two regions with one and two compsets.
    zpfbs.rebuild_two_phase_regions(Ttol=20)
    assert len(zpfbs.all_compsets) == 3
    assert len(zpfbs.two_phase_regions) == 2
    assert sorted([len(tpr.compsets) for tpr in zpfbs.two_phase_regions]) == [1, 2]


def test_zpf_boundary_sets_line_plot():
    """Test creating scatter plot LineCollections works"""
    compsets_298 = CompsetPair([
        BinaryCompset('P1', 298.15, 'B', 0.5, [0.5, 0.5]),
        BinaryCompset('P2', 298.15, 'B', 0.8, [0.2, 0.8]),
    ])

    compsets_300_diff_phases = CompsetPair([
        BinaryCompset('P2', 300, 'B', 0.5, [0.5, 0.5]),
        BinaryCompset('P3', 300, 'B', 0.8, [0.2, 0.8]),
    ])

    zpfbs = ZPFBoundarySets(['A', 'B'], v.X('B'))
    zpfbs.add_compsets(compsets_298)
    zpfbs.add_compsets(compsets_300_diff_phases)
    boundaries, tielines, legend = zpfbs.get_line_plot_boundaries()
    assert len(boundaries._paths) > 0
    assert len(tielines._paths) > 0


def test_zpf_boundary_set_scatter_plot():
    """Test creating scatter plot LineCollections works"""
    compsets_298 = CompsetPair([
        BinaryCompset('P1', 298.15, 'B', 0.5, [0.5, 0.5]),
        BinaryCompset('P2', 298.15, 'B', 0.8, [0.2, 0.8]),
    ])

    compsets_300_diff_phases = CompsetPair([
        BinaryCompset('P2', 300, 'B', 0.5, [0.5, 0.5]),
        BinaryCompset('P3', 300, 'B', 0.8, [0.2, 0.8]),
    ])

    zpfbs = ZPFBoundarySets(['A', 'B'], v.X('B'))
    zpfbs.add_compsets(compsets_298)
    zpfbs.add_compsets(compsets_300_diff_phases)
    boundaries, tielines, legend = zpfbs.get_scatter_plot_boundaries()
    x, y, col = boundaries['x'], boundaries['y'], boundaries['c']
    assert len(x) > 0
    assert len(x) == len(y)
    assert len(x) == len(col)
    assert len(tielines._paths) > 0
