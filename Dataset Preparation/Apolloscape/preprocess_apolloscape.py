# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:05:05 2021

@author: ANWOY
"""

import os
import numpy as np

"""---------------------------------
    APOLLOSCAPE DATASET FORMAT
    --------------------------------
    TRAINING/VALIDATION DATA:
    
    | frame_id | object_id | position_x | position_y | position_z | object_length | object_width | object_height | heading |
    
    TEST DATA:
    
    | frame_id | object_id | object_type | position_x | position_y | position_z | object_length | object_width | object_height | heading |

Target of pre-processing:
    frame_id: 1 to n
    object_id: 0 to N-1
"""

def generate_nparray_from_txt(file_path):
    '''
    to convert the data from the .txt file into one large numpy array
    :param file_path: path of the .txt file to load
    :return: numpy array of the data from txt file
    '''
    data = np.loadtxt(file_path)

    return data

def save_npy_to_file(data,save_path):
    '''
    to save the obtained merged numpy array to .npy
    :param data: large numpy array
    :param save_path: file path where this .npy file will be saved
    :return: None
    '''
    np.save(save_path, data, allow_pickle=True)
    
def load_npy_data(path_of_npy):
    '''
    to load the numpy array from .npy file
    :param path_of_npy: file path where the numpy array is saved
    :return: loaded numpy array
    '''
    data = np.load(path_of_npy)
    return data

def map_frame_id(first_col):
    '''
    to format the first column, i.e., all the time stamps are converted to 1 to maximum scale
    :param first_col: the first column of the merged numpy array
    :return: formatted first column of the merged numpy array, that is the frame ID's
    '''

    first_element = first_col[0]
    frame_ID_list = []
    new_id = 1

    for original_id in first_col:
        if original_id == first_element:
            frame_ID_list.append(new_id)
        else:
            first_element = original_id
            new_id+=1
            frame_ID_list.append(new_id)

    frame_ID_array = np.array(frame_ID_list, dtype= int).T

    return frame_ID_array

def map_object_id(data):
    '''
    to format the second column, i.e., object ID to start from 0 to maximum
    :param data: merged numpy array
    :return: formatted numpy array
    '''
    data_copy = data

    min_val = (np.amin(data_copy[:,1]))

    data_copy[:,1] = data_copy[:,1] - min_val

    return data_copy

def generate_data(file, data_type):

    '''
    to create the formatted data from the apolloscape data
    :param file: file in which the unformatted data is present
    :param data: whether it is train or val or test data
    :return: formatted data
    
    format of each row now:
        | frame_id | object_id | position_x | position_y | dataset_ID |
    '''
    dataset_ID = 4
    
    print("Processing..."+str(file))
    
    #Step 1: Generate a long numpy array from the text file
    data = generate_nparray_from_txt(file)
    
    #Step 2: Map the frame_id in first column to an integer between 1 to n
    first_col = data[:,0]
    new_first_col = map_frame_id(first_col)
    data[:,0] = new_first_col
    
    #Step 3: Map the object_id in second column to an integer between 0 to N-1
    data = map_object_id(data)
    
    
    #Delete the unnecessary columns in the data according to its type
    if data_type == 'train' or data_type == 'val':
        
        #DELETE position_z(4), object_length(5), object_width(6), object_height(7), heading(8) 
        formatted_data = np.delete(data,[4,5,6,7,8],axis=1)

    elif data_type == 'test':
        
        #DELETE object_type(2), position_z(5), object_length(6), object_width(7), object_height(8), heading(9) 
        formatted_data = np.delete(data,[2,5,6,7,8,9],axis=1)
    
    new_col = np.ones((formatted_data.shape[0] , 1) ) * dataset_ID 
    final_data = np.hstack((formatted_data, new_col))

    return final_data

def save_to_text(final_data, save_path, index):
    '''
    save the formatted array to a .txt file
    :param final_data: final formatted array
    :param save_path: file path where the .txt file has to be saved
    :param index: index value which will be used as a dataset ID
    :return: None
    ------------------------------------------------------------------
    format of each row after the data is saved in a text file:
        | dataset_ID | object_id | frame_id | position_x | position_y |
    '''
    data_list = np.ndarray.tolist(final_data)
    
    for item in data_list:
        #item[0]=frame_id
        item[0] = int(item[0])
        #item[1]=object_id
        item[1] = int(item[1])

    with open(save_path, 'w') as file:
        for l in data_list:
            # file.write('%d \t %d \t %f \t %f \t %f \n' %(l[0], l[1], l[2], l[3], l[4]))
            file.write("{},{},{},{},{}\n".format(index,l[1],l[0],l[2],l[3]))

def format_apolloscape_main(DATA_DIR, DATA_DIR_TEST, data_type, save_dir, file_no):

    train_val_file_names = []
    test_file_names = []

    for file in sorted(os.listdir(DATA_DIR)):
        if file.endswith('.txt'):
            train_val_file_names.append(file)


    for file in sorted(os.listdir(DATA_DIR_TEST)):
        if file.endswith('.txt'):
            test_file_names.append(file)

    if data_type == 'train':

        file = DATA_DIR +  train_val_file_names[int(file_no - 1)]

        generated_data = generate_data(file, data_type)

        save_path = save_dir + 'trainSet'+str(int(file_no))+'.npy'

        save_npy_to_file( generated_data, save_path)

    #     save_to_text(formatted_data, to_save_txt)


    if data_type == 'val':

        file = DATA_DIR + train_val_file_names[int(file_no - 1)]

        generated_data = generate_data(file, data_type)


        save_path = save_dir + 'valSet'+str(int(file_no-7))+'.npy'

        save_npy_to_file( generated_data, save_path)


    if data_type == 'test':

        file = DATA_DIR_TEST + test_file_names[int(file_no - 1)]

        generated_data = generate_data(file, data_type)


        save_path = save_dir + 'testSet'+str(int(file_no))+'.npy'

        save_npy_to_file( generated_data, save_path)

'''
Instructions:
1. Unzip the downloaded files sample_trajectory.zip and prediction_test.zip
2. Follow below format
DATA_DIR = folder_where_unzipped_apolloscape_data_is_present + '/sample_trajectory/asdt_sample_ trajectory/'
DATA_DIR_TEST = folder_where_unzipped_apolloscape_data_is_present + '/prediction_test/'


DATA_DIR = dir + 'sample_trajectory/asdt_sample_ trajectory/'
DATA_DIR_TEST = dir + '/prediction_test/'

data_type = 'val' #train, val, test

format_apolloscape_main(DATA_DIR, DATA_DIR_TEST, data_type, save_dir, file_no)
'''