import inkml
from graph import *
import trace
import math
from scipy.spatial import ConvexHull
import numpy as np
import parsing_feature_extractor as pfe
from detector import *

def get_distance_between_points(l1, l2):
    return math.sqrt((l1[0] - l2[0]) ** 2 +
                     (l1[1] - l2[1]) ** 2)

def get_line_of_sight_graph(list_of_symbols, inkml_obj, parsing_model):
    los = Graph(len(list_of_symbols))
    for sym in list_of_symbols:
        los.add_node(sym.id)

    for sym in list_of_symbols:
        xo, yo = sym.bbx, sym.bby
        blocked_intervals = []
        dist = {}
        for tg in list_of_symbols:
            if tg.id != sym.id:
                this_dist = get_distance_between_points([sym.bbx, sym.bby],
                                                   [tg.bbx, tg.bby])
                dist[tg] = this_dist
        dist = sorted(dist.items(), key=lambda x: x[1])
        for other_sym, distance in dist:
            theta_min = math.inf
            theta_max = -math.inf
            x = []
            y = []
            for tr in inkml_obj.get_traces_in_group(other_sym.id):
                x.extend(tr.x)
                y.extend(tr.y)
            xy = [[x[i], y[i]] for i in range(len(x))]
            for vert in ConvexHull(xy).vertices:
                w = np.array([xy[vert][0]- xo, xy[vert][1] - yo])
                h = np.array([1, 0])
                dr = get_distance_between_points(w, [0, 0])
                dr *= get_distance_between_points(h, [0, 0])
                theta = math.acos(np.dot(w, h.T) / dr)
                if xy[vert][1] < yo:
                    theta = math.radians(360) - theta

                theta_min = min(theta_min, theta)
                theta_max = max(theta_max, theta)

            put_in_blocked_view = 1
            for bv in blocked_intervals:
                if theta_min >= bv[0] and theta_max <= bv[1]:
                    put_in_blocked_view = 0
                    break
            if put_in_blocked_view == 1 and other_sym.id not in los.get_node(
                    sym.id).edge_l.keys():
                blocked_intervals.append([theta_min, theta_max])
                sym_trs = inkml_obj.get_traces_in_group(sym.id)
                other_sym_trs = inkml_obj.get_traces_in_group(other_sym.id)
                features = pfe.get_features(sym_trs, other_sym_trs)
                [r, w] = parsing_model.score_for_trace([features])
                if r != 'None':
                    los.add_edge(sym.id, other_sym.id, -1*w, label = r)
    return los


def main():
    file = '../TrainINKML/HAMEX/formulaire001-equation012.inkml'
    inkml_obj = inkml.marshal_inkml(file)
    inkml_obj.compute_all_tg_bb()
    parsing_model = Detector(training_features='features/'
                                               'training_set_features_for_parser_wlab.txt',
                                      training_gt='features/training_set_GT_for_parser_wlab.txt')

    parsing_model.deserialize_model_parameters(
        'training_parameters/parser_training_params.ds')
    g = get_line_of_sight_graph(inkml_obj.trace_groups, inkml_obj,
                                parsing_model)
    print('abc')

if __name__ == '__main__':
    main()



