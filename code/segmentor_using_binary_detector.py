import json
import time
import feature_extraction as fe
from detector import Detector
import segmentor_feature_extractor as sfe
from graph import *
from inkml import marshal_inkml

__author__ = ['Dharmendra Hingu', 'Sanjay Khatwani']


def can_these_traces_be_merged(trs, mst_a, inde, ids):
    """
    Checks if there is an edge in MST between two traces.
    """
    results = []
    for tr_i in range(len(trs)):
        for tr_j in range(len(trs)):
            if tr_i != tr_j:
                if (mst_a[inde[ids.index(trs[tr_i].id)]]
                    [inde[ids.index(trs[tr_j].id)]]) <= 0:
                    results.append(0)
                    break
        else:
            results.append(1)
    return sum(results) > 0


def get_confidence_of_traces_indexed(i, j, inkml_obj, detector, mst_of_inkml,
                                     inde, ids):
    """
    Gets a score for trace combinations (possible symbol)
    """
    traces = inkml_obj.traces
    gr = []
    for tr in range(len(traces)):
        if tr <= j and tr >= i:
            gr.append(traces[tr])

    probability = 0
    dec = 0
    if len(gr) and can_these_traces_be_merged(gr, mst_of_inkml, inde, ids):
        grouping_features = sfe.get_features(gr)
        [dec, probability] = detector.score_for_trace(
            [grouping_features])
    if dec == 1:
        return probability
    return 0


def compute_segmented_traces(n, detector, classifier, inkml_obj, mst, inde, ids):
    """
    Performs Dynamic Programming and produces S and Breaks arrays.
    """
    seg_ids = []
    for row in range(len(mst)):
        for col in range(len(mst[row])):
            if mst[row][col] > 0:
                features = sfe.get_features([inkml_obj.traces[row],
                                             inkml_obj.traces[col]])
                dec, prob = detector.score_for_trace([features])
                if dec == 1:
                    mst[row][col] = prob
                else:
                    mst[row][col] = 0

    for row in range(len(mst)):
        maxprob = max(mst[row])
        for col in range(len(mst[row])):
            if mst[row][col] < maxprob:
                mst[row][col] = 0
            else:
                mst[col][row] = maxprob
    segs = []
    visi = []
    for row in range(len(mst)):
        if row not in visi:
            seg = [inkml_obj.traces[row].id]
            for col in range(len(mst[row])):
                if mst[row][col] > 0:
                    seg.append(inkml_obj.traces[col].id)
                    visi.append(col)
            segs.append(seg)

    indices_to_remove = []
    for seg_i in range(len(segs)-2):
        seg_j = seg_i+1
        if len(list(set(segs[seg_i]).intersection(segs[seg_j]))) > 0:
            segs[seg_i].extend(list(set(segs[seg_j]) - set(segs[seg_i])))
            indices_to_remove.append(seg_j)

    new_segs = [segs[i] for i in range(len(segs)) if i not in indices_to_remove]
    return new_segs


def get_segmented_ids(kk, i, j):
    """
    Get ids of segments.
    """
    if kk[i][j] == j:
        return [list(range(i, j + 1))]
    else:
        first_half = get_segmented_ids(kk, i, kk[i][j])
        second_half = get_segmented_ids(kk, kk[i][j] + 1, j)
        first_half.extend(second_half)
        return first_half


def get_segmented_traces(segs, index_to_id):
    """
    Get trace objects of segments.
    """
    results = []
    for seg in segs:
        a = []
        for tr in seg:
            a.append(index_to_id[tr])
        results.append(a)
    return results


def get_segment_labels(segs, inkml_obj, detector):
    """
    Get classifications for segments.
    """
    labels = []
    for seg in segs:
        tracez = []
        for tr in seg:
            tracez.append(inkml_obj.get_trace(tr))
        features = fe.get_features(tracez)
        label, prob = detector.score_for_trace([features])
        if label == ',':
            label = 'COMMA'
        labels.append(label)
    return labels


def produce_lg(segs, labels, filename):
    """
    Generate lg file for segments and classes.
    """
    with open(filename, 'w') as f:
        for seg in range(len(segs)):
            line = "O, " + str(seg) + ", " + labels[seg] + ", 1.0, "
            for tr in range(len(segs[seg])):
                if tr < len(segs[seg]) - 1:
                    line += (segs[seg][tr] + ", ")
                else:
                    line += segs[seg][tr]
            line += '\n'
            f.write(line)


def generate_lg_file_for_inkml(file, classifier, detector=None,
                               model_name=None):
    """
    Processes one inkml file to produce lg file.
    Needs either detector object or location of model to use.
    """
    if model_name is None and detector is None:
        print("Either model_name or detector object is needed")
        return 0
    if detector is None:
        detector = Detector(model_name=model_name)

    inkml_obj = marshal_inkml(file)

    graph = inkml_obj.build_association_graph()
    mst, index_to_id = get_mst(graph)
    inde = list(index_to_id.keys())
    ids = list(index_to_id.values())

    seg_tr = compute_segmented_traces(len(inkml_obj.traces), detector,
                                      classifier,
                                      inkml_obj, mst, inde, ids)
    # seg_id = get_segmented_ids(breaks, 0, len(inkml_obj.traces) - 1)
    # seg_tr = get_segmented_traces(seg_id, index_to_id)
    labels = get_segment_labels(seg_tr, inkml_obj, classifier)
    filename = file.replace('.inkml', '.lg')
    produce_lg(seg_tr, labels, filename)
    return inkml_obj


def main(feature_files, model_files, file_list_of_files):
    detector = Detector(training_features=feature_files[0],
                        training_gt=feature_files[1])
    detector.deserialize_model_parameters(
        model_files[0])

    classifier = Detector(training_features=feature_files[2],
                        training_gt=feature_files[3])
    classifier.deserialize_model_parameters(model_files[1])


    with open(file_list_of_files, 'r') as file_pointer:
        file_list = json.load(file_pointer)

    for file_index in range(len(file_list)):
        # if file_index > 10:
        #     break
        file = file_list[file_index]
        print('Processing: ', file_index + 1, file)
        generate_lg_file_for_inkml(file, classifier, detector)


if __name__ == '__main__':
    feature_files = {
        'BINARY_DETECTOR_TRAINING_FEATURES':
            'features/binary_detector_training_set_features.txt',
        'BINARY_DETECTOR_TRAINING_GT':
            'features/binary_detector_training_set_GT.txt',
        'CLASSIFIER_BONUS_FEATURES':
            'features/classifier_bonus_features.txt',
        'CLASSIFIER_BONUS_GT': 'features/classifier_bonus_GT.txt',
        'CLASSIFIER_TRAINING_FEATURES':
            'features/classifier_training_set_features.txt',
        'CLASSIFIER_TRAINING_GT': 'features/classifier_training_set_GT.txt',
        'BINARY_DETECTOR_BONUS_FEATURES':
            'features/binary_detector_bonus_features.txt',
        'BINARY_DETECTOR_BONUS_GT': 'features/binary_detector_bonus_GT.txt',
        'PARSER_TRAINING_FEATURES': 'features/parser_training_set_features.txt',
        'PARSER_TRAINING_GT': 'features/parser_training_set_GT.txt',
        'PARSER_BONUS_FEATURES': 'features/parser_bonus_features.txt',
        'PARSER_BONUS_GT': 'features/parser_bonus_GT.txt'

    }

    model_files = {
        'CLASSIFIER': 'training_parameters/classifier_training_params.ds',
        'CLASSIFIER_BONUS': 'training_parameters/classifier_bonus_params.ds',
        'BINARY_DETECTOR': 'training_parameters/binary_detector_training_params'
                           '.ds',
        'PARSER': 'training_parameters/parser_training_params.ds',
        'BINARY_DETECTOR_BONUS':
            'training_parameters/binary_detector_bonus_training_params'
            '.ds',
        'PARSER_BONUS': 'training_parameters/parser_bonus_training_params.ds'
    }
    main(
        [feature_files['BINARY_DETECTOR_TRAINING_FEATURES'],
         feature_files['BINARY_DETECTOR_TRAINING_GT'],
         feature_files['CLASSIFIER_TRAINING_FEATURES'],
         feature_files['CLASSIFIER_TRAINING_GT']],
        [model_files['BINARY_DETECTOR'],
         model_files['CLASSIFIER']],
        'testing_files.txt'
    )
