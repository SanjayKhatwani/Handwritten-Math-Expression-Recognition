import json
from inkml import *
import numpy as np
from trace import trace
import math

__author__ = ['Dharmendra Hingu', 'Sanjay Khatwani']

def get_vertical_bb_distance(tr1, tr2):
    """
    Vertical bounding box distance
    """

    width = max(tr1.maxx, tr2.maxx) - min(tr1.minx, tr1.minx)
    height = max(tr1.maxy, tr2.maxy) - min(tr1.miny, tr2.miny)
    return round(float((tr1.bb_center_y - tr2.bb_center_y) / max(width,
                                                                height, 1)), 4)

def get_hor_bb_distance(tr1, tr2):
    """
    Horizontal bounding box distnace.
    """
    width = max(tr1.maxx, tr2.maxx) - min(tr1.minx, tr1.minx)
    height = max(tr1.maxy, tr2.maxy) - min(tr1.miny, tr2.miny)
    return round(float((tr1.bb_center_x - tr2.bb_center_x) / max(width,
                                                                height, 1)), 4)

def get_angle_between_bb(tr1, tr2):
    """
    Angle between bounding box centers and horizontal.
    :return:
    """
    width = max(tr1.maxx, tr2.maxx) - min(tr1.minx, tr1.minx)
    height = max(tr1.maxy, tr2.maxy) - min(tr1.miny, tr2.miny)
    angle = math.atan2(tr2.bb_center_y - tr1.bb_center_y,
                       tr2.bb_center_x - tr1.bb_center_x)
    angle = math.degrees(angle + 360) % 360
    angle = round(angle * 30) // 30
    return round(float(angle / max(width, height, 1)), 4)

def get_distance_between_bb_center(tr1, tr2):
    """
    Distance between bounding box centers.
    """
    width = max(tr1.maxx, tr2.maxx) - min(tr1.minx, tr1.minx)
    height = max(tr1.maxy, tr2.maxy) - min(tr1.miny, tr2.miny)
    dist = tr1.get_distance_bb_center(tr2)
    return round(float(dist / max(width, height, 1)), 4)

def get_offsets(traces):
    """
    Horizontal, vertical, top and bottom offsets between bounding box edges.
    """
    width = max(traces[0].maxx, traces[1].maxx) - min(traces[0].minx, traces[
        1].minx)
    height = max(traces[0].maxy, traces[1].maxy) - min(traces[0].miny, traces[
        1].miny)
    left_hor_offset = (traces[0].minx - traces[1].minx) / max(width, height, 1)
    right_hor_offset = (traces[0].maxx - traces[1].maxx) / max(width, height, 1)
    top_ver_offset = (traces[0].miny - traces[1].miny) / max(width, height, 1)
    bottom_ver_offset = (traces[0].maxy - traces[1].maxy) / max(width,
                                                                height, 1)

    return [round(left_hor_offset, 4), round(right_hor_offset, 4),
            round(top_ver_offset, 4), round(bottom_ver_offset, 4)]


def feature_corner_angle_with_center(tr1, tr2):
    """
    Angles between bounding box center of first symbol and corners of second
    symbol.
    """
    width = max(tr1.maxx, tr2.maxx) - min(tr1.minx, tr1.minx)
    height = max(tr1.maxy, tr2.maxy) - min(tr1.miny, tr2.miny)
    angle_list = []
    angle_list.append(angle_between_two_points([tr1.minx, tr1.miny],
                                               [tr2.bb_center_x,
                                                tr2.bb_center_y]))
    angle_list.append(angle_between_two_points([tr1.maxx, tr1.miny],
                                               [tr2.bb_center_x,
                                                tr2.bb_center_y]))
    angle_list.append(angle_between_two_points([tr1.minx, tr1.maxy],
                                               [tr2.bb_center_x,
                                                tr2.bb_center_y]))
    angle_list.append(angle_between_two_points([tr1.maxx, tr1.maxy],
                                               [tr2.bb_center_x,
                                                tr2.bb_center_y]))
    angle_list = [round(float(a / max(width, height, 1)), 4) for a in angle_list]

    return angle_list

def round_angle(angle,rnd):
    angle = math.degrees(angle + 360) % 360
    angle = round(angle * rnd) // rnd
    return angle

def angle_between_two_points(point1,point2):
    co = np.dot(point1,point2)
    sin = np.linalg.norm(np.cross(point1,point2))
    angle = np.arctan2(co, sin)
    angle = round_angle(angle,30)
    return angle

def feature_PSC(tr1, tr2):
    """
    Shape context features.
    """
    psc_features=[]
    minx = min(tr1.minx, tr2.minx)
    maxx = max(tr1.maxx, tr2.maxx)
    miny = min(tr1.miny, tr2.miny)
    maxy = max(tr1.maxy, tr2.maxy)
    bb_center = [minx + ((maxx - minx) / 2), miny + ((maxy - miny) / 2)]
    radius = minx if minx > maxx else maxx
    psc_features += compute_number_of_points_in_angle(bb_center, radius, tr1)
    psc_features += compute_number_of_points_in_angle(bb_center, radius, tr2)
    return psc_features

def distance(p1, p2):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    return np.sqrt(np.sum((p1-p2)**2))

def compute_number_of_points_in_angle(center, radius, tr):
    """
    Compute number of points in the span of an angle.
    """
    bins = np.zeros((6, 5))
    rad_round = radius/5
    count = 1
    for i in range(len(tr.x)):
        s1 = [tr.x[i], tr.y[i]]
        angle_center_bb = math.atan2(center[1] - s1[1], center[0] - s1[0])
        angle_center_bb = round_angle(angle_center_bb, 60)
        dist = round(distance(center, s1) * rad_round) / rad_round
        if dist < radius:
            count += 1
            d = int(round(dist / rad_round))
            a = angle_center_bb // 60
            if a == 6:
                a = 5
            bins[a][d-1] += 1
    return list(bins.flatten()/count)

def get_features(obj1_traces, obj2_traces):

    tr1 = trace()
    for tr in obj1_traces:
        tr1.x.extend(tr.x)
        tr1.y.extend(tr.y)
    tr1.calculate_centroids()
    tr1.id = 'O1'

    tr2 = trace()
    for tr in obj2_traces:
        tr2.x.extend(tr.x)
        tr2.y.extend(tr.y)
    tr2.calculate_centroids()
    tr2.id = 'O2'

    features = []

    features.append(get_vertical_bb_distance(tr1, tr2))
    features.append(get_hor_bb_distance(tr1, tr2))
    features.extend(feature_corner_angle_with_center(tr1, tr2))
    features.append(get_angle_between_bb(tr1, tr2))
    features.append(get_distance_between_bb_center(tr1, tr2))
    features.extend(get_offsets([tr1, tr2]))
    features.extend(feature_PSC(tr1, tr2))
    # traces = []
    # for tr in [tr1, tr2]:
    #     traces.append(tr.x)
    #     traces.append(tr.y)
    #
    # traces_dense = sfe.add_points_lin(traces)
    #
    # histogram_of_points = sfe.get_histogram_of_points(traces_dense, 5)
    # features.extend(histogram_of_points)
    return features

def main():
    # training files
    with open('testing_files.txt', 'r') as file_pointer:
        file_list = json.load(file_pointer)

    features_set = []
    ground_truth = []

    for file_index in range(len(file_list)):

        # if file_index > 10:
        #     break

        file = file_list[file_index]
        print('Extracting features for: ', file_index + 1, file)
        inkml_obj = marshal_inkml(file)
        lgfile = file.split('/')[-1].replace('.inkml', '.lg')
        lgfile = '../TrainINKML/training_gt_lg/'+lgfile
        marshal_objects_relations(lgfile, inkml_obj)

        rels_done = []
        for o in inkml_obj.objects:
            obj1_traces = inkml_obj.get_traces_in_object(o.id)
            rels = find_nodes_to_add_edges(o, inkml_obj)
            for rel in rels:
                obj2_traces = inkml_obj.get_traces_in_object(rel)
                features_set.append(get_features(obj1_traces, obj2_traces))
                relation_between_objs = inkml_obj.get_relation_between(
                    o.id.strip(), rel.strip())
                if(relation_between_objs[0] == None):
                    ground_truth.append('None')
                else:
                    ground_truth.append(relation_between_objs[0])
                    rels_done.append(relation_between_objs[1])

        for r in inkml_obj.relations:
            if r.id not in rels_done:
                obj1_traces = inkml_obj.get_traces_in_object(r.object_ids[0])
                obj2_traces = inkml_obj.get_traces_in_object(r.object_ids[1])
                features_set.append(get_features(obj1_traces, obj2_traces))
                ground_truth.append(r.label)
        # for r in inkml_obj.relations:
        #     obj1_traces = inkml_obj.get_traces_in_object(r.object_ids[0])
        #     obj2_traces = inkml_obj.get_traces_in_object(r.object_ids[1])
        #
        #     trss = order_objects(obj1_traces, obj2_traces)
        #     features_set.append(get_features(trss[0], trss[1]))
        #     ground_truth.append(r.label)


    with open('features/parser_training_set_features11.txt',
              'w') as \
            file_pointer:
        file_pointer.write(json.dumps(features_set))

    with open('features/parser_training_set_GT11.txt', 'w') as \
            file_pointer:
        file_pointer.write(json.dumps(ground_truth))

def order_objects(o1, o2):
    min1 = min([x.minx for x in o1])
    min2 = min([x.minx for x in o2])
    max1 = max([x.maxx for x in o1])
    max2 = max([x.maxx for x in o2])

    bb1 = min1 + ((max1 - min1) / 2)
    bb2 = min2 + ((max2 - min2) / 2)

    if bb2 > bb1:
        return [o1, o2]
    return [o2, o1]

def find_nodes_to_add_edges(this_tg, inkml_obj):
    if len(inkml_obj.objects) == 1:
        return []
    elif len(inkml_obj.objects) <= 3:
        return [i.id for i in inkml_obj.objects if i.id != this_tg.id]
    this_tg_tr = trace()
    for t in inkml_obj.get_traces_in_object(this_tg.id):
        this_tg_tr.x.extend(t.x)
        this_tg_tr.y.extend(t.y)
    this_tg_tr.calculate_centroids()
    this_tg_tr.id = "this"
    dist = {}
    for tg in range(len(inkml_obj.objects)):
        if inkml_obj.objects[tg].id != this_tg.id:
            other_tg_tr = trace()
            for t in inkml_obj.get_traces_in_object(inkml_obj.objects[tg].id):
                other_tg_tr.x.extend(t.x)
                other_tg_tr.y.extend(t.y)
            other_tg_tr.calculate_centroids()
            other_tg_tr.id = "Other"
            # if this_tg_tr.bb_center_x <= other_tg_tr.bb_center_x:
            this_dist = this_tg_tr.get_distance_bb_center(other_tg_tr)
            dist[inkml_obj.objects[tg].id] = this_dist
    dist = sorted(dist.items(), key=lambda x: x[1])
    if len(dist) >= 2:
        return [dist[0][0], dist[1][0]]
    elif len(dist) == 1:
        return [dist[0][0]]
    return []

if __name__ == '__main__':
    main()

