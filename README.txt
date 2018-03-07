Folder structure: 
code
...features
......ALL
.........X_train_ALL.npy
.........X_test_ALL.npy
.........Y_train_ALL.npy
.........Y_test_ALL.npy
......SYM
.........X_train_SYM.npy
.........X_test_SYM.npy
.........Y_train_SYM.npy
.........Y_test_SYM.npy
......BONUS
.........X_validation_BONUS.npy
...training_parameters
......gnb_params_bonus.sav
......gnb_params_sym.sav
......gnb_params.sav
......kd_tree_params_bonus.sav
......kd_tree_params_sym.sav
......kd_tree_params.sav
...generate_train_and_test_features.py
...feature_extraction.py
...model_train.py
...model_test.py
...junkTest2.csv
...junkTrain2.csv
...symbolTest2.csv
...sybmolTrain2.csv

The split is 70% training and 30% testing.
In 'features' directory we extract all the features for symbol and junk dataset. The ALL directory has features for symbol plus junk dataset.
The SYM directory has features for only symbols. The BONUS directory has features extracted for validation set that is provided. For this we use combined training and testing set of ALL directory as a training set. 

Once the model is trained we store the model parameters in training_parameters directory. This is serialized version of the model. If we just want to test/run the validation set we load this .sav file test it.

The assumption is that the dataset is present in the same directory as the code directory is.
generate_train_and_test_features.py is the driver file that extracts features from the given dataset. This file internally calls feature_extraction.py that focuses on extracting features for the given traces.
model_train.py is for training the models
model_test.py is for testing the models 

junkTest2.csv and junkTrain2.csv signifies the 30% and 70% testing and training set respectively
sybmbolTest2.csv and symbolTrain2.csv signifies the 30% and 70% testing and training set respectively. 

Sequence of commands
python3 generate_train_and_test_features.py
python3 model_train.py
python3 model_test.py

Prediction files are generated in the same directory as the code.
