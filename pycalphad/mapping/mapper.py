from pycalphad import Database, variables as v
from pycalphad.core.composition_set import CompositionSet
from pycalphad.mapping.primitives import STATEVARS, Point, Node, ZPFLine
from typing import List, Mapping, Tuple, Union
from pycalphad.mapping.map_strategies.step_strategy import StepStrategy
from pycalphad.mapping.map_strategies.tielines_strategy import TielineStrategy
from pycalphad.mapping.map_strategies.general_strategy import GeneralStrategy
import json
import numpy as np

"""
Context class containing system definition and the mapping strategy (step, tielines in planes, general)
"""

def _fix_conditions_list(conditions_str):
    conditions = {}
    for k in conditions_str:
        if k == "N":
            conditions[v.N] = conditions_str[k]
        elif k == "P":
            conditions[v.P] = conditions_str[k]
        elif k == "T":
            conditions[v.T] = conditions_str[k]
        elif "X" in k:
            conditions[v.X(k[2:])] = conditions_str[k]

    for k,var in conditions.items():
        if isinstance(var, list):
            conditions[k] = tuple(var)
    return conditions

def get_unique_elements(components):
    #Probably shouldn't keep vacancies in the list, but I'll have to go through everything and see what assumes that the components list has VA and fix
    return list(set(np.unique([[el for el in v.Species(c).constituents.keys()] for c in components])))

class Mapper():
    def __init__(self, database: Database, components : List[str], phases: List[str], conditions: Mapping[v.StateVariable, Union[float, Tuple[float, float, float]]], force_general: bool = False, force_tie_line: bool = False):
        # Get all axis conditions that are variable
        axis_vars = [k for k,val in conditions.items() if isinstance(val, tuple)]
        num_potential_conditions = sum([1 for av in axis_vars if av in STATEVARS])

        elements = get_unique_elements(components)

        stepping_mode = len(axis_vars) == 1
        is_binary_phase_diagram = (num_potential_conditions == 1) and (len(set(elements) - {"VA"}) == 2)
        is_ternary_isotherm = (num_potential_conditions == 0) and (len(set(elements) - {"VA"}) == 3)
        tielines_in_plane = (len(axis_vars) == 2) and (is_binary_phase_diagram or is_ternary_isotherm)

        if stepping_mode:
            self.strategy = StepStrategy(database, components, elements, phases, conditions)
        elif (tielines_in_plane and not force_general) or force_tie_line:
            self.strategy = TielineStrategy(database, components, elements, phases, conditions)
        else:
            self.strategy = GeneralStrategy(database, components, elements, phases, conditions)

    def iterate(self):
        return self.strategy.iterate()

    def do_map(self, iterations = -1):
        if iterations == -1:
            self.strategy.do_map()
            return
        else:
            finished = False
            while not finished:
                print("")
                for i in range(iterations):
                    finished = self.strategy.iterate()
                    if finished:
                        break
            return

    def save_map_data(self, dbfname, output_file: str):
        comps = self.strategy._system_definition["comps"]
        phases = self.strategy._system_definition["phases"]

        map_data = {}
        map_data["dbf"] = dbfname
        map_data["comps"] = comps
        map_data["phases"] = phases
        map_data["conditions"] = {str(k): v for k, v in self.strategy.conditions.items()}
        map_data["zpf"] = []
        for zpfline in self.strategy.zpf_lines:
            zpf_data = {}
            zpf_data["phases"] = []
            for cs in zpfline.points[0].fixed_composition_sets:
                zpf_data["phases"].append([cs.phase_record.phase_name, True])
            for cs in zpfline.points[0].free_composition_sets:
                zpf_data["phases"].append([cs.phase_record.phase_name, False])
            zpf_data["points"] = []
            for point in zpfline.points:
                point_data = [{str(k):v for k,v in point.global_conditions.items()}]
                for cs in point.stable_composition_sets:
                    point_data.append([cs.phase_record.phase_name, cs.NP, list(cs.dof)])
                zpf_data["points"].append(point_data)
            map_data["zpf"].append(zpf_data)
        map_data["nodes"] = []
        for node in self.strategy.node_queue.nodes:
            node_data = [{str(k):v for k,v in node.global_conditions.items()}]
            for cs in node.stable_composition_sets:
                node_data.append([cs.phase_record.phase_name, cs.NP, list(cs.dof)])
            map_data["nodes"].append(node_data)

        s = json.dumps(map_data)
        s = s.replace("{", "{\n\t")
        s = s.replace("}", "\n}")
        s = s.replace(', \"comps', ',\n\t\"comps')
        s = s.replace(', \"phases', ',\n\t\"phases')
        s = s.replace(', \"conditions', ',\n\t\"conditions')
        s = s.replace('\n\t\"P\"', '\"P\"')
        s = s.replace('\n}, \"zpf', '},\n\t\"zpf')
        s = s.replace(', \"nodes', ',\n\t\"nodes')
        s = s.replace(', \"points', ',\n\t\t\"points')
        s = s.replace('{\n\t\"phases', '{\n\t\t\"phases')
        s = s.replace('[{', '\n\t\t\t[{')
        s = s.replace('}, {', '\t\t},\n\t\t{')
        s = s.replace('\n}, [\"', '}, [\"')
        s = s.replace('[{\n\t\"', '[{\"')
        with open(output_file, 'w') as f:
            f.write(s)

    @classmethod
    def load_map_data(cls, input_file: str):
        with open(input_file, "r") as f:
            map_data = json.load(f)

        db = Database(map_data["dbf"])
        mapper = cls(db, map_data["comps"], map_data["phases"], _fix_conditions_list(map_data["conditions"]))
        zpf_lines = []
        for zpf_data in map_data["zpf"]:
            fixed_phases = []
            free_phases = []
            for ph in zpf_data["phases"]:
                if ph[1]:
                    fixed_phases.append(ph[0])
                else:
                    free_phases.append(ph[0])
                zpfline = ZPFLine(fixed_phases, free_phases)
            points = []
            for p in zpf_data["points"]:
                conditions = _fix_conditions_list(p[0])
                compsets = []
                for csdata in p[1:]:
                    cs = CompositionSet(mapper.strategy.phase_records[csdata[0]])
                    cs.update(np.array(csdata[2][len(STATEVARS):]), csdata[1], np.array(csdata[2][:len(STATEVARS)]))
                    if csdata[1] == 0:
                        cs.fixed = True
                    compsets.append(cs)
                points.append(Point(conditions, [cs for cs in compsets if cs.fixed], [cs for cs in compsets if not cs.fixed], []))
            zpfline.points = points
            zpf_lines.append(zpfline)
        mapper.strategy.zpf_lines = zpf_lines

        for node_data in map_data["nodes"]:
            conditions = _fix_conditions_list(node_data[0])
            compsets = []
            for csdata in node_data[1:]:
                cs = CompositionSet(mapper.strategy.phase_records[csdata[0]])
                cs.update(np.array(csdata[2][len(STATEVARS):]), csdata[1], np.array(csdata[2][:len(STATEVARS)]))
                if csdata[1] == 0:
                    cs.fixed = True
                compsets.append(cs)
            mapper.strategy.node_queue.add_node(Node(conditions, [cs for cs in compsets if cs.fixed], [cs for cs in compsets if not cs.fixed], [], None), True)
        return mapper
