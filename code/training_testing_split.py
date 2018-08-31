import json
from os import listdir
from os.path import isfile, join

from inkml import marshal_inkml

__author__ = ['Dharmendra Hingu', 'Sanjay Khatwani']


def get_file_listing():
    """
    Gets a list of all files in a folder.
    """
    file_directory = '../TrainINKML/'
    # get list of all the .inkml file names
    file_list = list()
    for directory in listdir(file_directory):
        if '.' not in directory:
            for file in listdir(join(file_directory, directory)):
                full_path = join(join(file_directory, directory), file)
                if isfile(full_path) and '.inkml' in full_path:
                    file_list.append(full_path)
    return file_list


def main():
    # file_list = get_file_listing()
    with open('testing_files.txt', 'r') as file_pointer:
        file_list = json.load(file_pointer)
    if isfile('symbol_count_te.txt'):
        with open('symbol_count.txt', 'r') as f:
            symbol_count = json.load(f)
        print(symbol_count)
    else:
        symbol_count = dict()
        for file_index in range(len(file_list)):
            file = file_list[file_index]
            print('Parsing', file)
            inkml_obj = marshal_inkml(file)
            for trace_group in inkml_obj.trace_groups:
                # print(trace_group.annotation_mml) # prints an individual symbol
                if symbol_count.get(trace_group.annotation_mml) is None:
                    symbol_count[trace_group.annotation_mml] = [1, 1]
                else:
                    symbol_count[trace_group.annotation_mml][0] += 1
                    symbol_count[trace_group.annotation_mml][1] += 1

        print('Symbol Counts:', symbol_count)

        with open('symbol_count_te.txt', 'w') as f:
            f.write(json.dumps(symbol_count))

            training_files = []
            testing_files = []

            for file_index in range(len(file_list)):
                current_symbol_count = dict()
                file = file_list[file_index]
                print('Training/Testing split:', file)
                inkml_obj = marshal_inkml(file)
                for trace_group in inkml_obj.trace_groups:
                    current_symbol_count[trace_group.annotation_mml] = current_symbol_count.get(trace_group.annotation_mml,
                                                                                                0) + 1
                tmp = dict()
                for k in current_symbol_count.keys():
                    tmp[k] = symbol_count[k][1]
                # print(current_symbol_count)
                for key in sorted(tmp, key=tmp.get):

                    # update current here

                    symbol_count = {key: [symbol_count[key][0], symbol_count[key][1] - current_symbol_count.get(key, 0)]
                                    for key in symbol_count.keys()}

                    if symbol_count[key][1] > symbol_count[key][0] * 0.3:
                        # put in training set
                        training_files.append(file)
                    else:
                        # put in testing set
                        testing_files.append(file)
                    break  # keep this break

            # training files print number and save
            print('Training_files', len(training_files))
            with open('training_files.txt', 'w') as f:
                f.write(json.dumps(training_files))

            # testing files print number and save
            print('Testing_files', len(testing_files))
            with open('testing_files.txt', 'w') as f:
                f.write(json.dumps(testing_files))


if __name__ == '__main__':
    main()  # setting the entry point
