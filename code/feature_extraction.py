import json
import time
import numpy as np
from scipy.misc import imresize

from inkml import marshal_inkml

__author__ = ['Dharmendra Hingu', 'Sanjay Khatwani']


def get_aspect_ratio(traces):
    """
    This method calculates the aspect ratio from the input sequence of traces
    aspect ratio = width / height
    width = xmax - xmin; height = ymax - ymin
    :param traces: Sequence of x and y coordinates of traces
    :return: aspect ratio
    """
    x_max = max([max(item) for item in traces[0::2]])
    y_max = max([max(item) for item in traces[1::2]])
    x_min = min([min(item) for item in traces[0::2]])
    y_min = min([min(item) for item in traces[1::2]])

    width = x_max - x_min
    height = y_max - y_min

    if width <= 0:
        width = 0.01
    if height <= 0:
        height = 0.01
    return width / height


def get_directions(traces):
    """
    This method counts the directions that traces in a symbol go through.
    1=Left, 2=Right, 3=Up, 4=Down
    :param traces: Sequence of x and y coordinates of traces
    :return: [Number of directions (min=1, max=4), First direction,
    Last direction]
    """
    directions = []
    for trace_xi in range(0, len(traces), 2):
        xs = traces[trace_xi]
        ys = traces[trace_xi + 1]
        # getAllX
        for x_i in range(1, len(xs)):
            if len(directions) < 4:
                # Left
                if 1 not in directions and (xs[x_i] - xs[x_i - 1]) < 0:
                    directions.append(1)
                # Right
                if 2 not in directions and (xs[x_i] - xs[x_i - 1]) > 0:
                    directions.append(2)
                # Up
                if 3 not in directions and (ys[x_i] - ys[x_i - 1]) > 0:
                    directions.append(3)
                # Down
                if 4 not in directions and (ys[x_i] - ys[x_i - 1]) < 0:
                    directions.append(4)
    return directions


def get_number_of_traces(traces):
    """
    This method returns the number of traces in a symbol
    :param traces: Sequence of x and y coordinates of traces
    :return: Number of traces.
    """
    return len(traces) / 2


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

    return hist_of_points.flatten().tolist()


def get_curve_information(traces):
    """
    This method calculates the average curvature in all the traces of a symbol.
    It calculates arctan((y[i+2] - y[i-2]) / (x[i+2] - x[i-2])) for every
    point i in every trace.
    Then it calculates and returns the average of all curvature information.
    This method has been implemented using the technique described in:

    H. Shu, "On-Line Handwriting Recognition Using Hidden Markov Models",
    Boston, 1997.

    :param traces: Sequence of x and y coordinates of traces
    :return: Average Curvature of symbol.
    """
    traces_curves = []
    for trace_xi in range(0, len(traces), 2):
        curves = []
        xs = traces[trace_xi]
        ys = traces[trace_xi + 1]
        # getAllX
        for x_i in range(1, len(xs)):
            if len(xs) > 4:
                if x_i < 2:
                    delta_x = xs[x_i + 2] - xs[x_i]
                    delta_y = ys[x_i + 2] - ys[x_i]
                elif x_i > len(xs) - 3:
                    delta_x = xs[x_i] - xs[x_i - 2]
                    delta_y = ys[x_i] - ys[x_i - 2]
                else:
                    delta_x = xs[x_i + 2] - xs[x_i - 2]
                    delta_y = ys[x_i + 2] - ys[x_i - 2]
                if delta_x != 0:
                    curves.append(np.arctan(delta_y / delta_x))
        if len(curves) == 0:
            traces_curves.append(0)
        else:
            traces_curves.append(np.mean(curves))
    return np.mean(traces_curves)


def det(a, b):
    """
    This method returns the determinant of lists a and b
    :param a: List
    :param b: List
    :return: Determinant of a and b
    """
    return abs(a[0] * b[1] - a[1] * b[0])


def do_these_lines_intersect(p1, p2, p3, p4, axis):
    """
    This method calculates the intersection point of the line defined by
    points p1 p2 and line defined by points p3 and p4.
    Then it checks if the intersection point lies on both the lines.
    :param p1: coordinates of first point on Line 1
    :param p2: coordinates of second point on Line 1
    :param p3: coordinates of first point on Line 1
    :param p4: coordinates of second point on Line 2
    :param axis: True of intersection point lies on both lines and False
    otherwise.
    :return:
    """
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    x3 = p3[0]
    y3 = p3[1]
    x4 = p4[0]
    y4 = p4[1]

    xdiff = [x2 - x1, x4 - x3]
    ydiff = [y2 - y1, y4 - y3]

    div = det(xdiff, ydiff)
    if div == 0:
        return False

    d = (det([x1, y1], [x2, y2]), det([x3, y3], [x4, y4]))
    x_intersection = det(d, xdiff) / div
    y_intersection = det(d, ydiff) / div

    if axis == 1:
        if x_intersection <= x2 and x_intersection >= x1 and x_intersection \
                <= x4 and x_intersection >= x3:
            return True
    else:
        if y_intersection <= y2 and y_intersection >= y1 and y_intersection \
                <= y4 and y_intersection >= y3:
            return True

    return False


def get_crossings_in_boundary(traces, xstart, xend, ystart, yend, axis):
    """
    This method assumes 9 lines passing through a boundary, which is alined
    to an axis defined by axis.
    Then it calculates the number of times a traces in a symbol intersects
    with any of the 9 lines from above.
    :param traces: Sequence of x and y coordinates of traces
    :param xstart: Starting x-coordinate of the boundary
    :param xend: Ending x-coordinate of the boundary
    :param ystart: Starting y-coordinate of the boundary
    :param yend: Ending y-coordinate of the boundary
    :param axis: Axis to which boundary is aligned 1=X, 2=Y.
    :return: Number of intersection.
    """
    count = 0
    if axis == 1:
        points = np.linspace(xstart, xend, 9)
    else:
        points = np.linspace(ystart, yend, 9)
    for trace_index in range(0, len(traces), 2):
        xs = traces[trace_index]
        ys = traces[trace_index + 1]
        for cor_index in range(len(xs) - 1):
            t_point1 = [xs[cor_index], ys[cor_index]]
            t_point2 = [xs[cor_index + 1], ys[cor_index + 1]]

            for point in points:
                if axis == 1:
                    point1 = [point, ystart]
                    point2 = [point, yend]
                else:
                    point1 = [xstart, point]
                    point2 = [xend, point]
                if do_these_lines_intersect(t_point1, t_point2, point1,
                                            point2, axis):
                    count += 1
    return count / 9


def get_crossing_feature(traces):
    """
    This method gets the crossing feature for traces in a symbol.
    It divdes the x span in 5 equal spans and then gets the
    crossings in each boundary defined by a span. The same thing is done for
    y span.
    Average crossings in each span is returned.

    This technique has been implemented as described in:

    K. Davila, S. Ludi and R. Zanibbi, "Using Off-line Features and Synthetic
    Data for On-line Handwritten Math Symbol Recognition",
    ICFHR, Crete, Greece, 201, 2014.


    :param traces: Sequence of x and y coordinates of traces
    :return: 10 average crossings in 10 regions.
    """
    x_crossings = []
    y_crossings = []
    x_max = max([max(item) for item in traces[0::2]])
    y_max = max([max(item) for item in traces[1::2]])
    x_min = min([min(item) for item in traces[0::2]])
    y_min = min([min(item) for item in traces[1::2]])

    x_lines = np.linspace(x_min, x_max, num=6)
    y_lines = np.linspace(y_min, y_max, num=6)

    for x_li in range(1, len(x_lines)):
        x_crossings.append(get_crossings_in_boundary(traces, x_lines[x_li - 1],
                                                     x_lines[x_li], y_min,
                                                     y_max, 1))

    for y_li in range(1, len(y_lines)):
        y_crossings.append(get_crossings_in_boundary(traces, x_min, x_max,
                                                     y_lines[y_li - 1],
                                                     y_lines[y_li], 2))

    return [x_crossings, y_crossings]


def get_features(trace_objects):
    """
    This method extracts features from a symbol.
    Following features are extracted:
    1. Number of directions
    2. First direction
    3. Last direction
    4. aspect ratio
    5. number of traces
    6 - 30. 2d Histogram 5*5
    31. Average curvature
    32 - 42. Crossings
    :param trace_objects: Sequence of x y coordinates in symbols, seperated in traces.
    :return: Features.
    """

    traces = []
    for tr in trace_objects:
        traces.append(tr.x)
        traces.append(tr.y)

    features = []
    traces_dense = add_points_lin(traces)
    traces_new_origin = shift_origin_to_zero_and_normalize(traces_dense)

    directions = get_directions(traces_new_origin)
    if len(directions) == 0:
        directions.append(0)
    features.append(len(directions))
    features.append(directions[0])
    features.append(directions[-1])

    aspect_ratio = get_aspect_ratio(traces_new_origin)
    features.append(aspect_ratio)

    number_of_traces = get_number_of_traces(traces_new_origin)
    features.append(number_of_traces)

    histogram_of_points = get_histogram_of_points(traces_new_origin, 5)
    features.extend(histogram_of_points)

    average_curve = get_curve_information(traces_new_origin)
    features.append(average_curve)

    crossings = get_crossing_feature(traces_new_origin)
    features.extend(crossings[0])
    features.extend(crossings[1])

    # rounding all to some fixed decimal points
    for index in range(len(features)):
        features[index] = float(round(features[index], 4))
    return features


def generate_matrix(traces):
    """
    This method converts a sequence of x, y coordinates to a binary matrix.
    :param traces: Sequence of x y coordinates in symbols, seperated in traces.
    :return: Binary matrix
    """
    x_max = max([max(item) for item in traces[0::2]])
    y_max = max([max(item) for item in traces[1::2]])
    matrix_dim = max(x_max, y_max) + 1
    matrix = np.zeros((matrix_dim, matrix_dim))
    for trace_xi in range(0, len(traces), 2):  # getAllX
        xs = traces[trace_xi]
        ys = traces[trace_xi + 1]
        for x_i in range(len(xs)):
            matrix[ys[x_i]][xs[x_i]] = 1
    matrix = fill_intermediate_points(matrix, traces)
    size = 100, 100
    new_binary_resize = imresize(matrix, size, interp='nearest')
    # visualize
    # plt.imshow(new_binary_resize, cmap='Greys',
    #            interpolation='nearest')
    # plt.show()

    return new_binary_resize.flatten().tolist()


def shift_origin_to_zero_and_normalize(traces):
    """
    This method shifts the origin of all symbols to zero.
    :param traces: Sequence of x y coordinates in symbols, separated in traces.
    :return: Traces shifted to zero
    """
    x_min = min([min(item) for item in traces[0::2]])
    y_min = min([min(item) for item in traces[1::2]])
    x_max = max([max(item) for item in traces[0::2]])
    y_max = max([max(item) for item in traces[1::2]])

    # x_max -= x_min
    # y_max -= y_min
    delta_y = y_max - y_min

    for x_or_y_index in range(len(traces)):
        for one_x_or_y_index in range(len(traces[x_or_y_index])):
            if x_or_y_index % 2 == 0:
                traces[x_or_y_index][one_x_or_y_index] -= x_min
                if delta_y > 0:
                    traces[x_or_y_index][one_x_or_y_index] /= delta_y
            else:
                if delta_y > 0:
                    traces[x_or_y_index][one_x_or_y_index] -= y_min
                    traces[x_or_y_index][one_x_or_y_index] /= delta_y
                else:
                    traces[x_or_y_index][one_x_or_y_index] = 0.5

    return traces


def fill_intermediate_points(traces):
    """
    This method inserts more 1s in a binary matrix to make it denser using
    linear interpolation.
    :param matrix: Binary matrix to dense.
    :param traces: Sequence of x y coordinates in symbols, seperated in traces.
    :return:
    """
    for trace_xi in range(0, len(traces), 2):  # getAllX
        xs = traces[trace_xi]
        ys = traces[trace_xi + 1]
        for x_i in range(1, len(xs)):
            x = xs[x_i]
            y = ys[x_i]
            x_p = xs[x_i - 1]
            y_p = ys[x_i - 1]
            diff_x = x - x_p
            diff_y = y - y_p
            if diff_x != 0:
                slope = 1.0 * diff_y / diff_x
                step = 1
                if diff_x > 5:
                    step = int(diff_x / 5)
                if diff_x < 0:
                    for start in range(int(diff_x), 0, step):
                        # matrix[int(round(slope * start + y_p))][x_p + start]= 1
                        traces[trace_xi].append(x_p + start)
                        traces[trace_xi + 1].append(slope * start + y_p)
                else:
                    for start in range(int(diff_x), 0, -1 * step):
                        # matrix[int(round(slope * start + y_p))][x_p + start]=1
                        traces[trace_xi].append(x_p + start)
                        traces[trace_xi + 1].append(slope * start + y_p)
    return traces


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
        start = time.time()
        for symbol in inkml_obj.trace_groups:

            features_set.append(
                get_features(inkml_obj.get_traces_in_group(symbol.id)))
            ground_truth.append(symbol.annotation_mml)



    with open('features/classifier_training_set_features11.txt', 'w') as \
            file_pointer:
        file_pointer.write(json.dumps(features_set))

    with open('features/classifier_training_set_GT11.txt', 'w') as file_pointer:
        file_pointer.write(json.dumps(ground_truth))

if __name__ == '__main__':
    main()
