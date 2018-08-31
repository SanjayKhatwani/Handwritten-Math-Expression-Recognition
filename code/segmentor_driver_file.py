import sys
import time

import segmentor_using_binary_detector as seg

__author__ = ['Dharmendra Hingu', 'Sanjay Khatwani']

"""
This program processes list of inkml files and produces output lg files.
"""

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
    if len(sys.argv) < 3:
        print("This file needs two arguments: path to file with a list of "
              "files, B/T. Where "
              "B/T = B for bonus set file and T for testing set file.")
    else:
        start = time.time()
        if str(sys.argv[2]) == 'T':
            seg.main(
                [feature_files['BINARY_DETECTOR_TRAINING_FEATURES'],
                 feature_files['BINARY_DETECTOR_TRAINING_GT'],
                 feature_files['CLASSIFIER_TRAINING_FEATURES'],
                 feature_files['CLASSIFIER_TRAINING_GT']],
                [model_files['BINARY_DETECTOR'],
                model_files['CLASSIFIER']],
                sys.argv[1].strip()
            )
        else:
            seg.main(
                [feature_files['BINARY_DETECTOR_BONUS_FEATURES'],
                 feature_files['BINARY_DETECTOR_BONUS_GT'],
                 feature_files['CLASSIFIER_BONUS_FEATURES'],
                 feature_files['CLASSIFIER_BONUS_GT']],
                [model_files['BINARY_DETECTOR_BONUS'],
                 model_files['CLASSIFIER_BONUS']],
                sys.argv[1].strip()
            )


        print("The predicted lg file is present in: ", str(sys.argv[1]).
              replace('.inkml', '.lg'))
        print("Time taken to process this file: ", time.time() - start,
              " seconds")
