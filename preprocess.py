from scipy import io as scio
import numpy as np
import os
dataset_path = {'seed4': 'eeg_feature_smooth', 'seed3': r'D:\dataset\MSMDA_way\SEED\ExtractedFeatures'}
def load_data(dataset_name):
    '''
    description: get all the data from one dataset
    param {type}
    return {type}:
        data: list 3(sessions) * 15(subjects), each data is x * 310
        label: list 3*15, x*1
    '''
    path, allmats = get_allmats_name(dataset_name)
    data = [([0] * 15) for i in range(3)]
    label = [([0] * 15) for i in range(3)]
    for i in range(len(allmats)):
        for j in range(len(allmats[0])):
            mat_path = path + '/' + str(i+1) + '/' + allmats[i][j]
            one_data, one_label = get_data_label_frommat(
                mat_path, dataset_name, i)
            data[i][j] = one_data.copy()
            label[i][j] = one_label.copy()
    return np.array(data), np.array(label)

def get_allmats_name(dataset_name):
    '''
    description: get the names of all the .mat files
    param {type}
    return {type}:
        allmats: list (3*15)
    '''
    path = dataset_path[dataset_name]
    sessions = os.listdir(path)
    sessions.sort()
    allmats = []
    for session in sessions:
        if session != '.DS_Store':
            mats = os.listdir(path + '/' + session)
            mats.sort()
            mats_list = []
            for mat in mats:
                mats_list.append(mat)
            allmats.append(mats_list)
    return path, allmats

def get_data_label_frommat(mat_path, dataset_name, session_id):
    '''
    description: load data from mat path and reshape to 851*310
    param {type}:
        mat_path: String
        session_id: int
    return {type}:
        one_sub_data, one_sub_label: array (851*310, 851*1)
    '''
    _, _, labels = get_number_of_label_n_trial(dataset_name)
    mat_data = scio.loadmat(mat_path)
    mat_de_data = {key: value for key,
                   value in mat_data.items() if key.startswith('de_LDS')}
    mat_de_data = list(mat_de_data.values())
    one_sub_data, one_sub_label = reshape_data(mat_de_data, labels[session_id])
    return one_sub_data, one_sub_label
def get_number_of_label_n_trial(dataset_name):
    '''
    description: get the number of categories, trial number and the corresponding labels
    param {type}
    return {type}:
        trial: int
        label: int
        label_xxx: list 3*15
    '''
    # global variables
    label_seed4 = [[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
                   [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2,
                       0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
                   [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]]
    label_seed3 = [[2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
                   [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
                   [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]]
    if dataset_name == 'seed3':
        label = 3
        trial = 15
        return trial, label, label_seed3
    elif dataset_name == 'seed4':
        label = 4
        trial = 24
        return trial, label, label_seed4
    else:
        print('Unexcepted dataset name')
def reshape_data(data, label):
    '''
    description: reshape data and initiate corresponding label vectors
    param {type}:
        data: list
        label: list
    return {type}:
        reshape_data: array, x*310
        reshape_label: array, x*1
    '''
    reshape_data = None
    reshape_label = None
    for i in range(len(data)):
        one_data = np.reshape(np.transpose(
            data[i], (1, 2, 0)), (-1, 310), order='F')
        one_label = np.full((one_data.shape[0], 1), label[i])
        if reshape_data is not None:
            reshape_data = np.vstack((reshape_data, one_data))
            reshape_label = np.vstack((reshape_label, one_label))
        else:
            reshape_data = one_data
            reshape_label = one_label
    return reshape_data, reshape_label