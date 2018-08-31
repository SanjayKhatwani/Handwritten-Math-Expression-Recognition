import json
import time
from os import makedirs
from os.path import exists

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

__author__ = ['Dharmendra Hingu', 'Sanjay Khatwani']


class Detector:
    __slots__ = ['x_train', 'x_test', 'y_train', 'y_test', 'clf']

    def __init__(self, training_features=None, training_gt=None,
                 testing_features=None,
                 testing_gt=None, model_name=None):
        x_train, x_test, y_train, y_test = self.read_file(training_features,
                                                          training_gt,
                                                          testing_features,
                                                          testing_gt)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        if model_name is not None:
            self.deserialize_model_parameters(model_name)

    def serialize_model_parameters(self, filename):
        """
        This method saves a model to file system to be used later.
        """

        if not exists('training_parameters'):
            makedirs('training_parameters')
        joblib.dump(self.clf, filename)

    def deserialize_model_parameters(self, file):
        """
        This method deserialize the model from the given file name
        """
        self.clf = joblib.load(file)

    def read_file(self, training_features, training_gt, testing_features,
                  testing_gt):
        """
        Read feature files.
        """
        x_train, x_test, y_train, y_test = [None, None, None, None]
        if training_features is not None:
            with open(training_features, 'r') as file_pointer:
                x_train = json.load(file_pointer)

        if training_gt is not None:
            with open(training_gt, 'r') as file_pointer:
                y_train = json.load(file_pointer)

        if testing_features is not None:
            with open(testing_features, 'r') as file_pointer:
                x_test = json.load(file_pointer)

        if testing_gt is not None:
            with open(testing_gt, 'r') as file_pointer:
                y_test = json.load(file_pointer)
        return [x_train, x_test, y_train, y_test]

    def random_forest_train(self):
        """
        Trains Random Forest. Saves the model to /training_parameters/
        """
        self.clf = RandomForestClassifier(max_depth=30, n_estimators=25)
        start = time.time()
        self.clf.fit(self.x_train, self.y_train)
        print("Time taken to train detector: ", time.time() - start, " seconds")

    def random_forest_test(self, trace=[]):
        if len(trace):
            return self.clf.predict_proba(trace)
        else:
            return self.clf.predict_proba(self.x_test)

    def score_for_trace(self, trace):
        """
        Returns a score for a symbol using saved Random Forest model.
        """
        class_ordering = self.clf.classes_
        probs = self.random_forest_test(trace)
        # print(probs)
        pairs = [[class_ordering[index], probs[0][index]] for index in
                 range(len(class_ordering))]
        res = sorted(pairs, key=lambda x: x[1], reverse=True)[0]
        return res


def main():
    detector = Detector('features/classifier_bonus_features'
                        '.txt',
                        'features/classifier_bonus_GT.txt')

    # when training
    detector.random_forest_train()
    detector.serialize_model_parameters(
        'training_parameters/classifier_bonus_training_params'
                           '.ds')

if __name__ == '__main__':
    main()  # setting the entry point
