import os
import pickle

def get_folder_names(directory):
    return sorted(name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)))

def get_file_names(directory):
    return [name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]

def load_label_encoder(path):
    with open(path, 'rb') as f:
        return pickle.load(f)