import csv
import xml.etree.ElementTree as ET
from os import listdir, remove
from os.path import isfile, join

from feature_extraction import *

__author__ = ['Sanjay Khatwani', 'Dharmendra Hingu']

# global variables to save the features into appropriate location
x_train_sym_file_name = "features/SYM/X_train_SYM.npy"  # only symbols training features
x_train_full_file_name = "features/ALL/X_train_ALL.npy"  # symbols plus junk training features

y_train_sym_file_name = "features/SYM/Y_train_SYM.npy"  # only symbols training classes
y_train_full_file_name = "features/ALL/Y_train_ALL.npy"  # symbols plus junk training classes

x_test_sym_file_name = "features/SYM/X_test_SYM.npy"  # only symbols testing features
x_test_full_file_name = "features/ALL/X_test_ALL.npy"  # symbols plus junk testing features

y_test_sym_file_name = "features/SYM/Y_test_SYM.npy"  # only symbols testing classes
y_test_full_file_name = "features/ALL/Y_test_ALL.npy"  # symbols plus junk testing classes

# list for all symbols file names
sym_file_names = [x_train_sym_file_name, y_train_sym_file_name,
                  x_test_sym_file_name, y_test_sym_file_name]

# list for symbols plus junk file names
all_file_names = [x_train_full_file_name, y_train_full_file_name,
                  x_test_full_file_name, y_test_full_file_name]

# store the features in these arrays
x_train = []
y_train = []
x_test = []
y_test = []
x_validation_bonus = []
y_validation_bonus = []  # at this point this will hold ids


def get_data_from_csv(file_name):
    """
    This function reads the csv file provided and construct name value pair dictionary where name corresponds
    to the unique id that is present in the .inkml file and value corresponds to the actual class
    :param file_name: name of the csv file
    :return: mapping of unique ids to actual class
    """
    name_class = {}
    with open(file_name, 'r') as csv_file:
        rows = csv.reader(csv_file, delimiter=',', quotechar='|')
        for row in rows:
            if len(row) > 0:
                name_class[row[0]] = row[1]
    return name_class


def extract_features_and_classes(file_directory, is_bonus=False, train_name_class_map=[],
                                 test_name_class_map=[]):
    """
    This function iterates over all the .inkml files present in the file_directory and extracts features for all the
    traces that we extract from .inkml file into appropriate array. To obtain the class from unique ids it makes use of
    map, which returns the class given the id.
    :param file_directory: location of all the training/testing .inkml files
    :param train_name_class_map: mapping of training unique ids vs class
    :param test_name_class_map: mapping of testing unique ids vs class
    :return: None
    """

    # get list of all the .inkml file names
    file_list = [f for f in listdir(file_directory) if
                 isfile(join(file_directory, f))]

    # for each .inkml file
    for file_index in range(len(file_list)):
        file = file_list[file_index]
        # if file_index > 5000:
        #      break

        if '.inkml' in file:  # ignore other files
            print(file_index + 1, 'Parsing', file)

            # use xml.etree packge to parse the inkml files
            tree = ET.parse(file_directory + file)
            root = tree.getroot()
            traces = []
            for trace in root.findall('{http://www.w3.org/2003/InkML}trace'):  # find all trace tags
                data = trace.text.rstrip().lstrip()
                xy = data.split(',')
                x = []
                y = []
                for one in xy:
                    one_xy = one.rstrip().lstrip().split(' ')
                    x.append(float(one_xy[0]))
                    y.append(float(one_xy[1]))
                traces.append(x)
                traces.append(y)

            extracted_feature = get_features(traces)  # extract features for all the traces
            annotation_id = ''

            for annotation in root.findall('{'
                                           'http://www.w3.org/2003/InkML}annotation'):  # find the unique ids
                if annotation.get('type') == 'UI':
                    annotation_id = annotation.text
            if not is_bonus:
                if train_name_class_map.get(annotation_id, None) is not None:
                    x_train.append(extracted_feature)  # training set
                    y_train.append([annotation_id,
                                    train_name_class_map[annotation_id]])

                else:
                    x_test.append(extracted_feature)  # testing set
                    y_test.append([annotation_id,
                                   test_name_class_map[annotation_id]])
            else:
                x_validation_bonus.append(extracted_feature)
                y_validation_bonus.append([annotation_id])


def save_all_files(filenames):
    """
    This function saves all the features into given file names
    :param filenames:
    :return:
    """
    try:
        # if the old files are present, purge it
        remove(filenames[0])
        remove(filenames[1])
        remove(filenames[2])
        remove(filenames[3])
    except OSError:
        pass
    np.save(filenames[0], np.array(x_train))
    np.save(filenames[1], np.array(y_train))
    np.save(filenames[2], np.array(x_test))
    np.save(filenames[3], np.array(y_test))


def main():
    """
    This is the main driver function.
    :return: None
    """

    # list of ground truth file names
    symbol_training_file = 'symbolTrain2.csv'
    symbol_testing_file = 'symbolTest2.csv'
    junk_training_file = 'junkTrain2.csv'
    junk_testing_file = 'junkTest2.csv'

    # generate the mapping of id vs class for all the ground truth file names
    junk_training_name_class = get_data_from_csv(junk_training_file)
    junk_testing_name_class = get_data_from_csv(junk_testing_file)
    symbol_training_name_class = get_data_from_csv(symbol_training_file)
    symbol_testing_name_class = get_data_from_csv(symbol_testing_file)

    # using the maps, extract the features and corresponding class for each .inkml files for all symbols
    extract_features_and_classes('../trainingSymbols/',
                                 train_name_class_map=symbol_training_name_class,
                                 test_name_class_map=symbol_testing_name_class)

    save_all_files(sym_file_names)  # this saves only sybmols

    # using the maps, extract the features and corresponding class for each .inkml files for junk
    extract_features_and_classes('../trainingJunk/',
                                 train_name_class_map=junk_training_name_class,
                                 test_name_class_map=junk_testing_name_class)

    save_all_files(all_file_names)  # this saves symbols + junk

    # using the maps, extract the features and corresponding class for each .inkml files for junk
    extract_features_and_classes('../testSymbols/', is_bonus=True)

    np.save('features/BONUS/X_validation_BONUS.npy', np.array(x_validation_bonus))
    np.save('features/BONUS/Y_validation_BONUS.npy', np.array(y_validation_bonus))


if __name__ == '__main__':
    main()
