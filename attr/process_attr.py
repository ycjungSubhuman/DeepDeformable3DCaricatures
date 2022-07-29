#!/usr/bin/env python3

"""
Find  3DCaricShop dataset's corresponding attribute labels from WebCariA dataset.
"""

import os
import pickle
import difflib


#----------------------------------------------------------------------------

def get_id_fname_dict_CaricShop(PATH_DIR):
    attr_dict = {}

    id = os.listdir(PATH_DIR)
    for id_name in id:
        id_path = os.path.join(PATH_DIR, id_name)
        attr_dict[id_name] = {}
        for f_name in os.listdir(id_path):
            f_name = os.path.splitext(f_name)[0]
            attr_dict[id_name][f_name] = []

    return attr_dict
        
#----------------------------------------------------------------------------

def get_label_WebCariA(PATH_DIR):
    f = open(os.path.join(PATH_DIR, "labels.txt"), "r")
    lines = f.readlines()

    attr_list = []
    for line in lines:
        attr_list.append(line.strip())
    
    return attr_list

#----------------------------------------------------------------------------

def get_attr_dict_WebCariA(PATH_DIR):
    f = open(os.path.join(PATH_DIR, "all_cari_data.txt"), "r")
    lines = f.readlines()

    data_dict = {}

    for line in lines:
        items = line.strip().split(' ')
        id_name, f_name = items[0].split('_')

        if not id_name in data_dict.keys():
            data_dict[id_name] = {}

        attr_list = []
        for item in items[1:]:
            attr_list.append(int(item))

        data_dict[id_name][f_name]= attr_list

    return data_dict

#----------------------------------------------------------------------------

def save_dict(PATH, data_dict):
    f = open(PATH, "wb")
    pickle.dump(data_dict, f)

#----------------------------------------------------------------------------

def load_dict(PATH):
    f = open(PATH, "rb")
    return pickle.load(f)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    caricshop_path = '/Kiwi/Data1/Dataset/3DCaricShop/processedData/rawMesh'
    WebCariA_path = '/Kiwi/Data1/Dataset/WebCariA_Dataset/WebCariA_Dataset/'
    data_path = './attr_data'

    '''
    data_dict = load_dict(os.path.join(data_path, 'data_dict.pkl'))

    attr_dict = get_id_fname_dict_CaricShop(caricshop_path)
    for id_name in attr_dict.keys():
        closest_name = difflib.get_close_matches(id_name, data_dict.keys())[0]
        for f_name in attr_dict[id_name].keys():
            attr_dict[id_name][f_name] = data_dict[closest_name][f_name]

    save_dict(os.path.join(data_path, 'attr_dict.pkl'), attr_dict)
    '''


    attr_list = get_label_WebCariA(WebCariA_path)
    save_dict('./attr_data/attr_list.pkl', attr_list)
    print(attr_list)
    close = difflib.get_close_matches('highcheekbones', attr_list)[0]
    print(close)
    
    index = attr_list.index(close)
    print(index)
