import os
import pickle

def get_folder_names(directory):
    try:
        folder_names = sorted(name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)))
        return folder_names
    except FileNotFoundError:
        print(f"디렉터리 '{directory}'를 찾을 수 없습니다.")
        return []
    except PermissionError:
        print(f"디렉터리 '{directory}'에 접근할 수 없습니다.")
        return []

# 폴더 안의 파일 이름 리스트
def get_file_names(directory):
    try:
        file_names = [name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]
        return file_names
    except FileNotFoundError:
        print(f"디렉터리 '{directory}'를 찾을 수 없습니다.")
        return []
    except PermissionError:
        print(f"디렉터리 '{directory}'에 접근할 수 없습니다.")
        return []

def load_label_encoder(path):
    with open(path, 'rb') as f:
        return pickle.load(f)