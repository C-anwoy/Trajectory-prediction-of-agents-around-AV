# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:51:22 2021

@author: ANWOY
"""

import json
import numpy as np

def load_sample(filepath):
    '''
    to load the sample.json file
    :param filepath: filepath of the sample.json
    :return: sample
    '''
    with open(filepath) as json_file:
        sample = json.load(json_file)

    return sample


def load_sample_data(filepath):
    '''
    to load the sample_data.json file
    :param filepath: filepath of the sample_data.json
    :return: sample_data
    '''
    with open(filepath) as json_file:
        sample_data = json.load(json_file)

    return sample_data


def load_sample_annotation(filepath):
    '''
    to load the sample_annotation.json file
    :param filepath: filepath of the sample_annotation.json
    :return: sample_annotation file
    '''
    with open(filepath) as json_file:
        sample_annotation = json.load(json_file)
        
    return sample_annotation


def get_token_timestamp_list(sample, scene_token):
    '''
    to get the list of sample_token-s and the corresponding timestamp for a particular scene
    :param sample: sample from sample.json
    :param scene_token: scene_token from scene.json
    :return: sorted list of [timestamp, sample_token] sub-lists
    '''

    token_timestamp_list = []
    visited_set = set()
    count = 0
    for items in sample:
        if items['scene_token'] == scene_token:
            token_timestamp_list.append([items['timestamp'],items['token']])
            visited_set.add(items['token'])
            count +=1

    sorted_list = sorted(token_timestamp_list)
    return sorted_list


def create_aggregated_list(sorted_token_timestamp_list, sample_data, sample_annotation):
    '''
    to create an aggregated list consisting of all the data that we require
    :param sorted_token_timestamp_list: sorted list of [timestamp, sample_token]
    :param sample_data: sample_data from sample_data.json
    :param sample_annotation: sample_annotation from from sample_annotation.json
    :return: aggr_list with all the required data sorted based on timestamp
    '''
    
    ################################################################################
    # This aggr_list consists of many dictionaries. Each dictionary has four keys: #
    #‘sample_token’ ,‘timestamp’,‘sample_data’, ‘annotation_data’                   #
    ################################################################################
    aggr_list = []
    
    new_dict = {}
    
    for token_timestamp in sorted_token_timestamp_list:
        new_dict['timestamp'] = token_timestamp[0]
        new_dict['sample_token'] = token_timestamp[1]
        new_dict['sample_data'] = {}
        aggr_list.append(new_dict)
        new_dict = {}

    for dictionary in aggr_list:
        sample_data = []
        for items in sample_data:
            if items['sample_token'] == dictionary['sample_token']:
                sample_data.append(items)

        dictionary['sample_data'] = sample_data

    for dictionary in aggr_list:
        annotation_data = []
        for items in sample_annotation:
            if items['sample_token'] == dictionary['sample_token']:
                annotation_data.append(items)

        dictionary['annotation_data'] = annotation_data

    return aggr_list


def get_timestamp_object_position_list(aggr_list):
    '''
    to obtain a list with [timestamp, object_id, position] sub-lists
    :param aggr_list: the aggregate list obtained in previous function
    :return: list of [timestamp, object_id, position] sub-lists and an aggregated instance_tokens_list
    '''
    the_req_list = []
    instance_tokens_list = []

    for dictionary in aggr_list:
        timestamp = dictionary['timestamp']

        for items in dictionary['annotation_data']:
            the_req_list.append([timestamp, items['instance_token'], items['translation']])
            instance_tokens_list.append(items['instance_token'])
            
    return the_req_list, instance_tokens_list

def object_id_mapping(instance_tokens_list):
    '''
    mapping each instance_token (or, object_id) with an integer value between 1 to N
    and obtain a dictionary whose keys are the actual values of instance_tokens and the
    corresponding value is an integer between 1 to N.
    
    :param instance_tokens_list: list of all instance_tokens
    :return: obtained mapping dictionary
    '''
    map_to_int = 1
    mapping_dictionary = {}
    visited_tokens_set = set()
    
    for object_id in instance_tokens_list:

        if object_id not in visited_tokens_set:
            mapping_dictionary[object_id] = map_to_int
            visited_tokens_set.add(object_id)
            map_to_int += 1
        else:
            continue

    return mapping_dictionary


def get_final_list(aggr_list, object_mapping_dictionary):
    '''
    final list with all [timestamp, object_id, position_x, position_y]
    :param aggr_list: aggregated list that was created in function 'create_aggregated_list'
    :param object_mapping_dictionary: mapping dictionary obtained in function 'object_id_mapping' 
    :return: list in the format [timestamp/frame_id, object_id, position_x, position_y]
    '''
    final_list = []

    for dictionary in aggr_list:
        timestamp = dictionary['timestamp']

        for items in dictionary['annotation_data']:
            final_list.append([timestamp, object_mapping_dictionary[items['instance_token']], items['translation'][0], items['translation'][1]])


    return final_list


def get_frame_id_list(final_list):
    '''
    to get the list of all frame_ids/timestamps
    :param final_list:  list in the format [timestamp, object_id, position_x, position_y]
    :return: array of all frame_id's
    '''
    frame_id_list = []
    
    for sub_list in frame_id_list:
        timestamp = sub_list[0]
        frame_id_list.append(timestamp)

    frame_id_arr = np.array(frame_id_list).T

    return frame_id_arr

def map_frame_id(first_col):
    '''
    to format the timestamp column
    param first_col: the first column to be formatted, timestamp_column
    :return: formatted column as an array
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

def format_lyft_main(dir):
    '''
    main function to format the lyft dataset
    :param dir: directory where dataset is present
    :return: pre-processed data
    '''

    ##STEP 1: Load all required json files
    
    #load sample.json
    sample_filepath = dir + 'sample.json'
    sample = load_sample(sample_filepath)
    
    #load sample_data.json
    sample_data_filepath = dir + 'sample_data.json'
    sample_data = load_sample_data(sample_data_filepath)
    
    #load sample_annotation.json
    sample_annotation_filepath = dir + 'sample_annotation.json'
    sample_annotation = load_sample_annotation(sample_annotation_filepath)

    #load scene.json
    scene_file_path = dir + 'scene.json'
    with open(scene_file_path) as json_file:
        scene = json.load(json_file)

    print('All required json files are loaded successfully...')

    ##STEP 2: Apply the pre-processing steps by iterating through all the samples/frames in a scene
    index = 1
    
    empty_array_train = np.empty((0,5))
    empty_array_test = np.empty((0,5))
    empty_array_val = np.empty((0,5))


    for items in scene:

        # using scene token to get the sample tokens
        scene_token = items['token']
        
        #sorted list of [timestamp, sample_token] sub-lists
        sorted_timestamp_list = get_token_timestamp_list(sample, scene_token)
        
        #get aggr_list with all the required data sorted based on timestamp
        aggr_list = create_aggregated_list(sorted_timestamp_list, sample_data, sample_annotation)
        
        #get list of [timestamp, object_id, position] sub-lists and an aggregated instance_tokens_list
        ts_ann_list, instance_tokens_list = get_timestamp_object_position_list(aggr_list)
        #obtain mapping dictionary for object_id mapping
        instance_matching_dict = object_id_mapping(instance_tokens_list)
        
        #get final list with sub-lists in the format [timestamp/frame_id, object_id, position_x, position_y]
        final_list = get_final_list(aggr_list, instance_matching_dict)
        
        #get the list of all frame_ids/timestamps
        frame_ID_arr = get_frame_id_list(final_list)
        #map frame_ids
        mapped_frame_id_col = map_frame_id(frame_ID_arr)
        
        #convert final_list to numpy array
        np_final_list = np.array(final_list)
        #replace first column with new mapped frame_id column
        np_final_list[:,0] = mapped_frame_id_col
        
        #add a dataset_ID for later recognition
        dataset_ID = 2
        new_col = np.ones((np_final_list[0] , 1) ) * dataset_ID 
        np_final_array = np.hstack((np_final_list, new_col))
        
        #ts_obj_ID_arr[:,4] = np.ones((ts_obj_ID_arr.shape[0],)) * index
        
        ##Splitting into train, validation and test sets
        if index >=0 and index <=126:
            empty_array_train =  np.concatenate((empty_array_train, np_final_array))  

        elif index >126 and index <= 144:
            empty_array_test =  np.concatenate((empty_array_test, np_final_array))  

        elif index> 144 and index <=180:
            empty_array_val =  np.concatenate((empty_array_val, np_final_array))  

        print("Completed the pre-processing of Scene " + index)
        index += 1

    return empty_array_train, empty_array_test, empty_array_val


DATA_DIR = 'directory/' ## provide the directory where the downloaded data is present

# train_dir = './resources/data/' + 'LYFT/train/*.txt'
files_path_to_sv_train = './resources/data/' + 'LYFT/train/trainSet0.npy'


# test_dir = './resources/data/' + 'LYFT/test/*.txt'
files_path_to_sv_test = './resources/data/' + 'LYFT/test/testSet0.npy'

# val_dir = './resources/data/' + 'LYFT/val/*.txt'
files_path_to_sv_val = './resources/data/' + 'LYFT/val/valSet0.npy'

print(files_path_to_sv_train)
print(files_path_to_sv_test)
print(files_path_to_sv_val)

tr, te, va  = format_lyft_main(DATA_DIR)

np.save(files_path_to_sv_train, tr)
np.save(files_path_to_sv_test, te)
np.save(files_path_to_sv_val, va)