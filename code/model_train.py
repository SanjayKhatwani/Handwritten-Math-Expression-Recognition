from os import makedirs
from os.path import exists

import numpy as np
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KDTree

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


def serialize_model_parameters(model, filename):
    """
    This method saves a model to file system to be used later.
    :param model: Model object
    :param filename: Name of file.
    :return:
    """

    if not exists('training_parameters'):
        makedirs('training_parameters')
    joblib.dump(model, filename)


def train_kd_tree(train_data, filename):
    """
    This method trains a kd-tree
    :param train_data: The data to train on
    :param filename: Name if file to save the kd-tree to.
    :return: N/a
    """
    np.random.seed(3)
    tree = KDTree(train_data)
    print('kd-tree Training Complete')
    serialize_model_parameters(tree, filename)
    print('kd-tree is serialized')


def train_gnb(x_train, y_train, filename):
    gnb = GaussianNB()
    naive_bayes = gnb.fit(x_train, y_train[:, 1])
    print('GNB Training Complete')
    serialize_model_parameters(naive_bayes, filename)
    print('GNB is serialized')


def load_data(x_train_file, y_train_file, x_test_file, y_test_file):
    """
    This method loads the features and class labels to be used for training.
    :param x_train_file: Name of training features file
    :param y_train_file: Name of training class labels file
    :param x_test_file: Name of testing features file
    :param y_test_file: Name of testing class labels file
    :return:
    """
    x_train = np.load(x_train_file)
    y_train = np.load(y_train_file)
    x_test = np.load(x_test_file)
    y_test = np.load(y_test_file)
    return [x_train, y_train, x_test, y_test]


def main(filenames, kd_model_name, gnb_model_name):
    """
    Main method.
    :param filenames: File names to load data from (Only symbols or symbols+junk)
    :return:
    """

    [x_train, y_train, x_test, y_test] = load_data(filenames[0], filenames[1],
                                                   filenames[2], filenames[3])

    print('Begin Training kd-tree')
    train_kd_tree(x_train, kd_model_name)

    print('Begin Training gnb')
    train_gnb(x_train, y_train, gnb_model_name)


def bonus_train():
    [x_train, y_train, x_test, y_test] = load_data(all_file_names[0], all_file_names[1],
                                                   all_file_names[2], all_file_names[3])
    x_train_bonus = np.concatenate(
        (x_train, x_test),
        axis=0
    )

    y_train_bonus = np.concatenate(
        (y_train, y_test),
        axis=0
    )

    train_kd_tree(x_train, 'training_parameters/kd_tree_params_bonus.sav')
    train_gnb(x_train, y_train, 'training_parameters/gnb_params_bonus.sav')


if __name__ == '__main__':
    print('Training only valid symbols')
    main(sym_file_names, 'training_parameters/kd_tree_params_sym.sav', 'training_parameters/gnb_params_sym.sav')
    print()

    print('Training valid symbols + junk')
    main(all_file_names, 'training_parameters/kd_tree_params_all.sav', 'training_parameters/gnb_params_all.sav')
    print()

    print('Training for bonus')
    bonus_train()
    print()
