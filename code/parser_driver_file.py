import sys
import symbol_level_parser as sym
import stroke_level_parser as str


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
        'CLASSIFIER' : 'training_parameters/classifier_training_params.ds',
        'CLASSIFIER_BONUS': 'training_parameters/classifier_bonus_params.ds',
        'BINARY_DETECTOR': 'training_parameters/binary_detector_training_params'
                           '.ds',
        'PARSER': 'training_parameters/parser_training_params.ds',
        'BINARY_DETECTOR_BONUS':
            'training_parameters/binary_detector_bonus_training_params'
                           '.ds',
        'PARSER_BONUS' : 'training_parameters/parser_bonus_training_params.ds'
    }


    if len(sys.argv) < 4:
        print("This file needs three arguments: path to file that contains the "
              "list of files to process, B/T. Where "
              "B/T = B for bonus set file and T for testing set file, 1/2. 1 "
              "for Symbol level parser and 2 for Stroke level parser.")
    else:
        if int(sys.argv[3].strip()) == 1:
            if sys.argv[2].strip() == 'T':
                sym.main(
                    [feature_files['PARSER_TRAINING_FEATURES'],
                    feature_files['PARSER_TRAINING_GT']],
                    [model_files['PARSER']],
                    sys.argv[1].strip()
                )
            else:
                sym.main(
                    [feature_files['PARSER_BONUS_FEATURES'],
                     feature_files['PARSER_BONUS_GT']],
                    [model_files['PARSER_BONUS']],
                    sys.argv[1].strip()
                )
        else:
            if sys.argv[2].strip() == 'T':
                str.main(
                    [feature_files['PARSER_TRAINING_FEATURES'],
                        feature_files['PARSER_TRAINING_GT'],
                        feature_files['BINARY_DETECTOR_TRAINING_FEATURES'],
                        feature_files['BINARY_DETECTOR_TRAINING_GT'],
                        feature_files['CLASSIFIER_TRAINING_FEATURES'],
                        feature_files['CLASSIFIER_TRAINING_GT']],
                    [model_files['PARSER'],
                     model_files['BINARY_DETECTOR'],
                     model_files['CLASSIFIER']],
                    sys.argv[1].strip()
                )
            else:
                str.main(
                    [feature_files['PARSER_BONUS_FEATURES'],
                     feature_files['PARSER_BONUS_GT'],
                     feature_files['BINARY_DETECTOR_BONUS_FEATURES'],
                     feature_files['BINARY_DETECTOR_BONUS_GT'],
                     feature_files['CLASSIFIER_BONUS_FEATURES'],
                     feature_files['CLASSIFIER_BONUS_GT']],
                    [model_files['PARSER_BONUS'],
                     model_files['BINARY_DETECTOR_BONUS'],
                     model_files['CLASSIFIER_BONUS']],
                    sys.argv[1].strip()
                )
    print('The output label graph files will be available in the same '
          'locations as the '
          'input inkml files with the same name and .lg extension')