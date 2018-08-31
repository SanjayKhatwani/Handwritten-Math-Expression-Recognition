import json
from graph import *
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt
from inkml import *
import math

__author__ = ['Dharmendra Hingu', 'Sanjay Khatwani']

def add_points_lin(traces):
    """
    This function finds 6 equidistant points between every pair of consecutive points using
    linear interpolation.
    :param traces: traces of points
    :return:
    """
    new_traces = []
    for trace_xi in range(0, len(traces), 2):
        xs = traces[trace_xi]
        ys = traces[trace_xi + 1]
        trace_x = [xs[0]]
        trace_y = [ys[0]]
        for x_i in range(1, len(xs)):
            x = xs[x_i]
            y = ys[x_i]
            x_p = xs[x_i - 1]
            y_p = ys[x_i - 1]
            if x_p < x:
                x_lin = np.linspace(x_p, x, 8)
                x_f = [x_p, x]
                y_f = [y_p, y]
            else:
                x_lin = np.linspace(x, x_p, 8)
                x_f = [x, x_p]
                y_f = [y, y_p]

            y_lin = np.interp(x_lin, x_f, y_f)
            trace_x.extend(x_lin[1:].tolist())
            trace_y.extend(y_lin[1:].tolist())
        new_traces.append(trace_x)
        new_traces.append(trace_y)
    return new_traces


def get_features(trace_objects):
    features = []

    features.append(get_dist_between_bb_centers(trace_objects))
    features.extend(get_bb_overlap(trace_objects))
    features.append(get_dist_between_centroids(trace_objects))
    features.append(get_first_point_dist(trace_objects))
    features.append(get_last_point_dist(trace_objects))
    features.extend(get_offsets(trace_objects))

    traces = []
    for tr in trace_objects:
        traces.append(tr.x)
        traces.append(tr.y)

    traces_dense = add_points_lin(traces)

    histogram_of_points = get_histogram_of_points(traces_dense, 5)
    features.extend(histogram_of_points)

    return features

def get_dist_between_bb_centers(traces):
    '''
    Distance between bounding box centers.
    '''
    width = max(traces[0].maxx, traces[1].maxx) - min(traces[0].minx, traces[
        1].minx)
    height = max(traces[0].maxy, traces[1].maxy) - min(traces[0].miny, traces[
        1].miny)
    mindist = 10000
    for tri in traces:
        for trj in traces:
            if tri.id != trj.id:
                dist = tri.get_distance_bb_center(trj)
                if dist < mindist:
                    mindist = dist
    return round(float(mindist / max(width, height, 1)), 4)

def get_dist_between_centroids(traces):
    """
    Distance between centroids of 2 symbols.
    """
    width = max(traces[0].maxx, traces[1].maxx) - min(traces[0].minx, traces[
        1].minx)
    height = max(traces[0].maxy, traces[1].maxy) - min(traces[0].miny, traces[
        1].miny)
    mindist = 10000
    for tri in traces:
        for trj in traces:
            if tri.id != trj.id:
                dist = tri.get_distance_centroid(trj)
                if dist < mindist:
                    mindist = dist
    return round(float(mindist / max(width, height, 1)), 4)

def get_bb_overlap(traces):
    """
    Overlap between two bounding boxes.
    """
    width = max(traces[0].maxx, traces[1].maxx) - min(traces[0].minx, traces[
        1].minx)
    height = max(traces[0].maxy, traces[1].maxy) - min(traces[0].miny, traces[
        1].miny)
    maxhoverlap = 0
    maxvoverlap = 0
    for tri in range(len(traces)):
        for trj in range(tri+1, len(traces)):
            if traces[tri].id != traces[trj].id and \
                traces[tri].maxx - traces[trj].minx > maxhoverlap:
                maxhoverlap = traces[tri].maxx - traces[trj].minx
            if traces[tri].id != traces[trj].id and traces[tri].maxy - traces[trj].miny\
                    > maxvoverlap:
                maxvoverlap = traces[tri].maxy - traces[trj].miny
    return [round(float(maxhoverlap / max(width, height, 1)), 4), round(float(
        maxvoverlap / max(width, height, 1)), 4)]

def get_first_point_dist(traces):
    width = max(traces[0].maxx, traces[1].maxx) - min(traces[0].minx, traces[
        1].minx)
    height = max(traces[0].maxy, traces[1].maxy) - min(traces[0].miny, traces[
        1].miny)
    maxdist = 0
    for tri in traces:
        for trj in traces:
            dist = math.sqrt((tri.x[0] - trj.x[0]) ** 2 + (
                    tri.y[0] - trj.y[0]) ** 2)
            if dist > maxdist:
                maxdist = dist
    return round(float(maxdist / max(width, height, 1)))

def get_last_point_dist(traces):
    width = max(traces[0].maxx, traces[1].maxx) - min(traces[0].minx, traces[
        1].minx)
    height = max(traces[0].maxy, traces[1].maxy) - min(traces[0].miny, traces[
        1].miny)
    maxdist = 0
    for tri in traces:
        for trj in traces:
            dist = math.sqrt((tri.x[-1] - trj.x[-1]) ** 2 + (
                    tri.y[-1] - trj.y[-1]) ** 2)
            if dist > maxdist:
                maxdist = dist
    return round(float(maxdist / max(width, height, 1)))

def get_offsets(traces):
    """
    Get offsets between horizontal, vertical, top and bottom edges of the
    bounding boxes.
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



def get_histogram_of_points(traces, number_of_bins):
    """
    This method calculates a 2-d histogram of points. This histogram is of
    size number_of_bins * number_of_bins
    :param traces: Sequence of x and y coordinates of traces
    :param number_of_bins: Size of one axis on the 2-d histogram
    :return:
    """

    x_max = max([max(item) for item in traces[0::2]])
    y_max = max([max(item) for item in traces[1::2]])
    x_min = min([min(item) for item in traces[0::2]])
    y_min = min([min(item) for item in traces[1::2]])

    hist_of_points = np.zeros((number_of_bins, number_of_bins))
    x_array = np.linspace(x_min, x_max, number_of_bins + 1)
    y_array = np.linspace(y_max, y_min, number_of_bins + 1)

    for trace_xi in range(0, len(traces), 2):
        xs = traces[trace_xi]
        ys = traces[trace_xi + 1]
        # getAllX
        for x_i in range(1, len(xs)):
            x = xs[x_i]
            y = ys[x_i]

            for bin_index in range(number_of_bins):
                if bin_index == number_of_bins - 1:
                    index_x = bin_index
                    break
                elif x >= x_array[bin_index] and x < x_array[bin_index + 1]:
                    index_x = bin_index
                    break

            for bin_index in range(number_of_bins):
                if bin_index == number_of_bins - 1:
                    index_y = bin_index
                    break
                elif y <= y_array[bin_index] and y > y_array[bin_index + 1]:
                    index_y = bin_index
                    break
            hist_of_points[index_x][index_y] += 1

    hist_of_points = hist_of_points.flatten().tolist()
    total = sum(hist_of_points)
    if total > 0:
        hist_of_points_norm = [round((i / total), 4) for i in hist_of_points]
        return hist_of_points_norm

    return hist_of_points

def generate_matrix(traces):
    """
    This method converts a sequence of x, y coordinates to a binary matrix.
    :param traces: Sequence of x y coordinates in symbols, seperated in traces.
    :return: Binary matrix
    """
    matrix = np.zeros((100, 100))
    for trace_xi in range(0, len(traces), 2):  # getAllX
        xs = traces[trace_xi]
        ys = traces[trace_xi + 1]
        for x_i in range(len(xs)):
            i = int(round(ys[x_i], 2) * 100)-1
            j = int(round(xs[x_i], 2) * 100)-1
            if i > 99: i= 99
            if j > 99: j= 99
            matrix[i][j] = 1
    size = 100, 100
    new_binary_resize = imresize(matrix, size, interp='nearest')
    # visualize
    plt.imshow(new_binary_resize, cmap='Greys',
               interpolation='nearest')
    plt.show()

def main():
    # training files
    with open('testing_files.txt', 'r') as file_pointer:
        file_list = json.load(file_pointer)

    features_set = []
    ground_truth = []
    for file_index in range(len(file_list)):
        file = file_list[file_index]
        print('Extracting features for: ', file_index + 1, file)
        inkml_obj = marshal_inkml(file)

        graph = inkml_obj.build_association_graph()
        mst, index_to_id = get_mst(graph)
        for row in range(len(mst)):
            for col in range(row+1, len(mst[row])):
                if mst[row][col] > 0:
                    tr1 = inkml_obj.get_trace(index_to_id[row])
                    tr2 = inkml_obj.get_trace(index_to_id[col])

                    if tr1.tgid != -1 or tr2.tgid != -1:
                        features_set.append(get_features([tr1, tr2]))
                        if tr1.tgid == -1 or tr2.tgid == -1:
                            ground_truth.append(0)
                        elif tr1.tgid == tr2.tgid:
                            ground_truth.append(1)
                        else:
                            ground_truth.append(0)


    with open('features/binary_detector_features.txt',
              'w') as \
            file_pointer:
        file_pointer.write(json.dumps(features_set))

    with open('features/binary_detector_GT.txt', 'w') as \
            file_pointer:
        file_pointer.write(json.dumps(ground_truth))


if __name__ == '__main__':
    main()