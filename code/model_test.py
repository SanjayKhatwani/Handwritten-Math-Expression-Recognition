import numpy as np
from sklearn.externals import joblib

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


def deserialize_model(filename):
    """
    This file deserialize the model from the given file name
    :param filename: name of the file
    :return: model object
    """
    model = joblib.load(filename)
    return model


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


def test_kd_tree(x_test, y_train, filename, top_k_classes):
    """
    This method tests the kd-tree
    :param x_test: data to test on
    :param y_train: for class label references
    :param filename: to retrieve serialized version of the model
    :param top_k_classes: get top k neighbors
    :return: y_hat_test: top 10 unique prediction for each of the test sample
    """
    tree = deserialize_model(filename)
    y_hat_test = []

    dist, ind = tree.query(x_test, k=top_k_classes)

    for row in range(len(x_test)):
        y_hat_test.append([])
        for index in ind[row]:
            if y_train[index][1] not in y_hat_test[row]:
                y_hat_test[row].append(y_train[index][1])
            if len(y_hat_test[row]) == 10:
                break
    print(y_hat_test)
    return y_hat_test


def test_gnb(x_test, y_train, filename):
    """
    This function tests the Gaussian Naive Bayes classifier
    :param x_test: testing or validation data
    :param y_train: ignored
    :param filename: to retrieve serialized version of the model
    :return: top 10 unique prediction for each of the test sample
    """
    gnb = deserialize_model(filename)
    predictions = gnb.predict_proba(x_test)
    y_class_order = gnb.classes_  # order of classes
    # print([[i] for i in gnb.predict(x_test)]) # gives the single prediction for the sample test
    y_hat_test = []
    # print(y_class_order)
    for prediction in predictions:
        tmp = []
        for index in range(len(prediction)):
            tmp.append([prediction[index], y_class_order[index]])

        tmp = sorted(tmp, key=lambda x: x[0], reverse=True)[:10]
        y_hat_test.append(np.array(tmp)[:, 1])
    # print(y_hat_test)
    return y_hat_test


def prepare_for_evaluation2(y_test, y_hat_test, model_name):
    """
    This function prepares the text file for the evaluation
    :param y_test: actual give ground truth for the testing set
    :param y_hat_test:  predicted values
    :param model_name:  for the filename construction
    :return:
    """
    # print(y_test)
    # print(y_hat_test)

    for index in range(len(y_hat_test)):
        y_hat_test[index].insert(0, y_test[index, 0])

    # result = np.concatenate((np.reshape(y_test[:, 0], (len(y_test), 1)),
    #                          np.array([np.array(pred) for pred in y_hat_test])),
    #                         axis=1)

    with open(model_name + '_predictions.txt', 'w') as file_pointer:
        for line in y_hat_test:
            line = ','.join(line)
            file_pointer.write(line + '\n')


def prepare_for_evaluation(y_test, y_hat_test, model_name):
    """
    This function prepares the text file for the evaluation
    :param y_test: actual give ground truth for the testing set
    :param y_hat_test:  predicted values
    :param model_name:  for the filename construction
    :return:
    """
    # print(y_test)
    # print(y_hat_test)
    result = np.concatenate((np.reshape(y_test[:, 0], (len(y_test), 1)),
                             np.array([np.array(pred) for pred in y_hat_test])),
                            axis=1)

    with open(model_name, 'w') as file_pointer:
        for line in result:
            line = ','.join(line.tolist())
            file_pointer.write(line + '\n')


def bonus_test():
    [x_test, y_test, y_train] = load_data(all_file_names[2], all_file_names[3],
                                          all_file_names[1])

    y_train_bonus = np.concatenate(
        (y_train, y_test),
        axis=0
    )
    x_validation_bonus = np.load('features/BONUS/X_validation_BONUS.npy')
    y_validation_bonus = np.load('features/BONUS/Y_validation_BONUS.npy')

    # use these if doing all_file_names
    y_hat_test = test_kd_tree(x_validation_bonus, y_train_bonus, 'training_parameters/kd_tree_params_bonus.sav', 10)
    prepare_for_evaluation(y_validation_bonus, y_hat_test, '../kd_bonus')

    y_hat_test = test_gnb(x_validation_bonus, y_train_bonus, 'training_parameters/gnb_params_bonus.sav')
    prepare_for_evaluation(y_validation_bonus, y_hat_test, '../gnb_bonus')


def main(filenames, kd_model_name, gnb_model_name):
    [x_train, y_train, x_test, y_test] = load_data(all_file_names[0], all_file_names[1],
                                                   all_file_names[2], all_file_names[3])
    if len(y_test):
        # use these if doing sym_file_names

        # training part
        y_hat_test = test_kd_tree(x_train, y_train, kd_model_name, 100)
        prepare_for_evaluation(y_train, y_hat_test, '../kdtree-output-train-' + kd_model_name + '.txt')

        y_hat_test = test_gnb(x_train, y_train, gnb_model_name)
        prepare_for_evaluation(y_train, y_hat_test, '../gnb-output-train-' + gnb_model_name + '.txt')

        # testing part
        y_hat_test = test_kd_tree(x_test, y_train, kd_model_name, 100)
        prepare_for_evaluation(y_test, y_hat_test, '../kdtree-output-test-' + kd_model_name + '.txt')

        y_hat_test = test_gnb(x_test, y_train, gnb_model_name)
        prepare_for_evaluation(y_test, y_hat_test, '../gnb-output-test-' + gnb_model_name + '.txt')

        # # use these if doing all_file_names
        # training part
        # y_hat_test = test_kd_tree(x_train, y_train, 'training_parameters/kd_tree_params_all.sav', 100)
        # prepare_for_evaluation2(y_train, y_hat_test, '../kd_train')

        # y_hat_test = test_gnb(x_train, y_train, 'training_parameters/gnb_params_all.sav')
        # prepare_for_evaluation(y_train, y_hat_test, '../gnb_train')

        # testing part
        # y_hat_test = test_kd_tree(x_test, y_train, 'training_parameters/kd_tree_params_all.sav', 100)
        # prepare_for_evaluation2(y_test, y_hat_test, '../kd')
        #
        # y_hat_test = test_gnb(x_test, y_train, 'training_parameters/gnb_params_all.sav')
        # prepare_for_evaluation(y_test, y_hat_test, '../gnb')

    else:
        print('No testing data found...')


if __name__ == '__main__':
    print('Evaluating model for only valid symbols')
    main(sym_file_names, 'training_parameters/kd_tree_params_sym.sav', 'training_parameters/gnb_params_sym.sav')
    print()

    print('Evaluating model for valid symbols + junk')
    main(all_file_names, 'training_parameters/kd_tree_params_all.sav', 'training_parameters/gnb_params_all.sav')
    print()

    print('Evaluating model for bonus')
    bonus_test()
    print()
