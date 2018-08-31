parser_driver_file.py is the main program to run the parser. The paths to input inkml files need to provided as a list. This list is provided in text file.

$ parser_driver_file <path to text file with paths to input inkmls> <B/T> <1/2>

The output label graphs are stored in the same location as the input inkml files.

There are three parameters to this program. The first parameter is the path to the text file which has paths to all the input inkml files. These paths need to be present as a list of strings. Following is an example:

["../bonus_inkml_symbols/18_em_0.inkml", "../bonus_inkml_symbols/18_em_1.inkml", "../bonus_inkml_symbols/18_em_10.inkml", "../bonus_inkml_symbols/18_em_11.inkml", "../bonus_inkml_symbols/18_em_12.inkml",.........]

The second parameter is B/T which specifies which models need to be used: Bonus model or the Testing models.
The third parameter is '1' for symbol level parser and '2' for the stroke level parser.

All the features and ground truths are present isn src/features folder. 
All the training parameters are present in the src/training_parameters.

symbol_level_parser.py runs individualy to process the test set. The output lg files are stored in the same location as the input inkml files.
stroke_level_parser.py runs individually to process the test set. The output lg files are stored in the same location as the input inkml files.

parser_driver_file.py uses the stroke_level_parser.py and stroke_level_parser.py files. stroke_level_parser.py uses the segmentator_using_binary_detector.py.

segmentator_using_binary_detector.py contains the new segmentator that uses a binary detector that provides the merge/split decision for a pair of nodes.

training_set.txt, testing_set.txt, stroke_bonus_set.txt and symbol_bonus_set.txt contain the lists for files in training set, testing set, bonus for stroke level parser and bonus for symbol level parser respectively.

Segmenter uses the binary_detector params and features.

Project3
...src
......features
......training_parameters
......segmentor_using_binary_detector.py
......parser_driver_file.py
......stroke_level_parser.py
......symbol_level_parser.py
...TrainINKML
......expressmatch
......extension
......HAMEX
......KAIST
......MathBrush
......MfrDB

