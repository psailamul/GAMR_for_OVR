"""
    MyCustomsDataset
    > HEPT4MD_ImageDataset
    How to call:
        from ImageDataset import HEPT4MD_ImageDataset
    #exec(open("ImageDataset.py").read())     
"""


##########################################################################	
##########################################################################	
import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


########################################################################## 
# Lite-Ver --- for HP Search 
##########################################################################  1
class HEPT4MD_ImageDataset(Dataset):
    def __init__(self, dict_of_exp_info, transform=None, internal_shuffle_flag = False, use_glob_return = False):
        self.internal_shuffle_flag = internal_shuffle_flag #not use because it's better to set it once at DataLoaders
        self.use_glob_return = use_glob_return
        self.task_name = dict_of_exp_info['task_name']  #'SD', 'LR', 'AB'
        self.obj_config = dict_of_exp_info['obj_config'] # 'SA' , 'ID', 'LOC'
        self.num_distractor = dict_of_exp_info['num_distractor'] # 0, 1, 2, 3
        self.set_type = dict_of_exp_info['set_type'] #train, test, validation
        
        #Get paths and labels
        dict_of_objects = self.get_all_image_paths_and_labels(dict_of_exp_info, internal_shuffle_flag)
        self.path_to_folder = dict_of_objects['path_to_folder']
        self.image_files = dict_of_objects['image_files']
        self.one_task_label = dict_of_objects['one_task_label']
        self.labels_by_class_numbers = dict_of_objects['labels_by_class_numbers']
        self.total_samples = dict_of_objects['total_samples']

        self.transform = transform

    def get_all_image_paths_and_labels(self, dict_of_exp_info, use_glob_return):

        from info_scripts.info_dataset import get_path_to_folder, list_of_files_in_data_folder, get_label_one_task_only, transform_class_labels_to_number

        task_name = dict_of_exp_info['task_name'] #'SD', 'LR', 'AB'
        obj_config = dict_of_exp_info['obj_config']
        num_distractor = dict_of_exp_info['num_distractor']
        set_type = dict_of_exp_info['set_type'] #train, test, validation
        path_to_folder = get_path_to_folder(task_name, obj_config, num_distractor, set_type)
        dict_of_lists = list_of_files_in_data_folder(path_to_folder)
        total_samples = dict_of_lists['total_samples']
        if use_glob_return: #if True: use whatever glob return -- no-reorder
            image_files = dict_of_lists['list_of_all_files']
            all_labels = dict_of_lists['labels_from_file']
        else: #if not shuffle -- order by sample ID
            image_files = dict_of_lists['filenames']
            all_labels = dict_of_lists['labels']
        # get class label for one task only
        one_task_label = get_label_one_task_only(all_labels, task_name)
        return_dict = transform_class_labels_to_number(one_task_label)
        labels_by_class_numbers = return_dict['all_class_number']
        num_classes = return_dict['num_classes']
        all_classes = return_dict['all_classes']
        map_label_to_num = return_dict['map_label_to_num']
        #Return 
        dict_of_objects = {
            'path_to_folder':path_to_folder,
            'image_files':image_files,
            'one_task_label':one_task_label,
            'labels_by_class_numbers':labels_by_class_numbers,
            'total_samples':total_samples
        }
        return dict_of_objects

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        img_path = os.path.join(self.path_to_folder, self.image_files[idx])
        image = read_image(img_path)
        label = self.labels_by_class_numbers[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


########################################################################## 
# General-Ver --- for real exp
########################################################################## 

class HEPT4MD_CustomizeDataset(Dataset):
    def __init__(self, dict_of_exp_info, transform=None, internal_shuffle_flag = False):
        self.internal_shuffle_flag = internal_shuffle_flag
        self.task_name = dict_of_exp_info['task_name']  #'SD', 'LR', 'AB'
        self.obj_config = dict_of_exp_info['obj_config'] # 'SA' , 'ID', 'LOC'
        self.num_distractor = dict_of_exp_info['num_distractor'] # 0, 1, 2, 3
        self.set_type = dict_of_exp_info['set_type'] #train, test, validation
        
        #Get paths and labels
        dict_of_objects = self.get_all_image_paths_and_labels(dict_of_exp_info, internal_shuffle_flag)
        self.path_to_folder = dict_of_objects['path_to_folder']
        self.image_files = dict_of_objects['image_files']
        self.one_task_label = dict_of_objects['one_task_label']
        self.labels_by_class_numbers = dict_of_objects['labels_by_class_numbers']
        self.total_samples = dict_of_objects['total_samples']

        self.transform = transform

    def get_all_image_paths_and_labels(self, dict_of_exp_info, internal_shuffle_flag):

        from info_scripts.info_dataset import return_get_path_to_folder_function, list_of_files_in_data_folder, get_label_one_task_only, transform_class_labels_to_number
        
        get_path_to_folder = return_get_path_to_folder_function(RUNNING_MODE = 'GENERAL')

        task_name = dict_of_exp_info['task_name'] #'SD', 'LR', 'AB'
        obj_config = dict_of_exp_info['obj_config']
        num_distractor = dict_of_exp_info['num_distractor']
        set_type = dict_of_exp_info['set_type'] #train, test, validation
        path_to_folder = get_path_to_folder(task_name, obj_config, num_distractor, set_type)
        dict_of_lists = list_of_files_in_data_folder(path_to_folder)
        total_samples = dict_of_lists['total_samples']
        if internal_shuffle_flag: #if shuffle -- get whatever glob return
            image_files = dict_of_lists['list_of_all_files']
            all_labels = dict_of_lists['labels_from_file']
        else: #if not shuffle -- order by sample ID
            image_files = dict_of_lists['filenames']
            all_labels = dict_of_lists['labels']
        # get class label for one task only
        one_task_label = get_label_one_task_only(all_labels, task_name)
        return_dict = transform_class_labels_to_number(one_task_label)
        labels_by_class_numbers = return_dict['all_class_number']
        num_classes = return_dict['num_classes']
        all_classes = return_dict['all_classes']
        map_label_to_num = return_dict['map_label_to_num']
        #Return 
        dict_of_objects = {
            'path_to_folder':path_to_folder,
            'image_files':image_files,
            'one_task_label':one_task_label,
            'labels_by_class_numbers':labels_by_class_numbers,
            'total_samples':total_samples
        }
        return dict_of_objects

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        img_path = os.path.join(self.path_to_folder, self.image_files[idx])
        image = read_image(img_path)
        label = self.labels_by_class_numbers[idx]
        if self.transform:
            image = self.transform(image)
        return image, label




########################################################################## 
# More Generic Version -- can specify the folder path seed number
########################################################################## 

class HEPT4MD_CustomizeDataset_genVer(Dataset):
    def __init__(self, dict_of_exp_info, dataset_seed =12345, transform=None, internal_shuffle_flag = False):
        self.dataset_seed = dataset_seed
        self.internal_shuffle_flag = internal_shuffle_flag
        self.task_name = dict_of_exp_info['task_name']  #'SD', 'LR', 'AB'
        self.obj_config = dict_of_exp_info['obj_config'] # 'SA' , 'ID', 'LOC'
        self.num_distractor = dict_of_exp_info['num_distractor'] # 0, 1, 2, 3
        self.set_type = dict_of_exp_info['set_type'] #train, test, validation
        
        #Get paths and labels
        dict_of_objects = self.get_all_image_paths_and_labels(dict_of_exp_info, internal_shuffle_flag)
        self.path_to_folder = dict_of_objects['path_to_folder']
        self.image_files = dict_of_objects['image_files']
        self.one_task_label = dict_of_objects['one_task_label']
        self.labels_by_class_numbers = dict_of_objects['labels_by_class_numbers']
        self.total_samples = dict_of_objects['total_samples']

        self.transform = transform

    def get_all_image_paths_and_labels(self, dict_of_exp_info, internal_shuffle_flag):

        from info_scripts.info_dataset import return_get_path_to_folder_function, list_of_files_in_data_folder, get_label_one_task_only, transform_class_labels_to_number
        
        get_path_to_folder = return_get_path_to_folder_function(RUNNING_MODE = 'SEED')
        dataset_seed = self.dataset_seed
        task_name = dict_of_exp_info['task_name'] #'SD', 'LR', 'AB'
        obj_config = dict_of_exp_info['obj_config']
        num_distractor = dict_of_exp_info['num_distractor']
        set_type = dict_of_exp_info['set_type'] #train, test, validation
        path_to_folder = get_path_to_folder(dataset_seed, task_name, obj_config, num_distractor, set_type)
        dict_of_lists = list_of_files_in_data_folder(path_to_folder)
        total_samples = dict_of_lists['total_samples']
        if internal_shuffle_flag: #if shuffle -- get whatever glob return
            image_files = dict_of_lists['list_of_all_files']
            all_labels = dict_of_lists['labels_from_file']
        else: #if not shuffle -- order by sample ID
            image_files = dict_of_lists['filenames']
            all_labels = dict_of_lists['labels']
        # get class label for one task only
        one_task_label = get_label_one_task_only(all_labels, task_name)
        return_dict = transform_class_labels_to_number(one_task_label)
        labels_by_class_numbers = return_dict['all_class_number']
        num_classes = return_dict['num_classes']
        all_classes = return_dict['all_classes']
        map_label_to_num = return_dict['map_label_to_num']
        #Return 
        dict_of_objects = {
            'path_to_folder':path_to_folder,
            'image_files':image_files,
            'one_task_label':one_task_label,
            'labels_by_class_numbers':labels_by_class_numbers,
            'total_samples':total_samples
        }
        return dict_of_objects

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        img_path = os.path.join(self.path_to_folder, self.image_files[idx])
        image = read_image(img_path)
        label = self.labels_by_class_numbers[idx]
        if self.transform:
            image = self.transform(image)
        return image, label        

########################################################################## 
# More Generic Version -- can specify the folder path seed number
########################################################################## 

class HEPT4MD_CustomizeDataset_POLY(Dataset):
    def __init__(self, dict_of_exp_info, poly_set ='DECO', transform=None, internal_shuffle_flag = False):
        self.poly_set = poly_set #'DECO', 'TETRO'
        self.internal_shuffle_flag = internal_shuffle_flag
        self.task_name = dict_of_exp_info['task_name']  #'SD', 'LR', 'AB'
        self.obj_config = dict_of_exp_info['obj_config'] # 'SA' , 'ID', 'LOC'
        self.num_distractor = dict_of_exp_info['num_distractor'] # 0, 1, 2, 3
        self.set_type = dict_of_exp_info['set_type'] #train, test, validation
        
        #Get paths and labels
        dict_of_objects = self.get_all_image_paths_and_labels(dict_of_exp_info, internal_shuffle_flag)
        self.path_to_folder = dict_of_objects['path_to_folder']
        self.image_files = dict_of_objects['image_files']
        self.one_task_label = dict_of_objects['one_task_label']
        self.labels_by_class_numbers = dict_of_objects['labels_by_class_numbers']
        self.total_samples = dict_of_objects['total_samples']

        self.transform = transform

    def get_all_image_paths_and_labels(self, dict_of_exp_info, internal_shuffle_flag):

        from info_scripts.info_dataset import return_get_path_to_folder_function, list_of_files_in_data_folder, get_label_one_task_only, transform_class_labels_to_number
        
        get_path_to_folder = return_get_path_to_folder_function(RUNNING_MODE = 'POLY')
        poly_set = self.poly_set
        task_name = dict_of_exp_info['task_name'] #'SD', 'LR', 'AB'
        obj_config = dict_of_exp_info['obj_config']
        num_distractor = dict_of_exp_info['num_distractor']
        set_type = dict_of_exp_info['set_type'] #train, test, validation
        path_to_folder = get_path_to_folder(task_name, obj_config, num_distractor, set_type, poly_set)
        dict_of_lists = list_of_files_in_data_folder(path_to_folder)
        total_samples = dict_of_lists['total_samples']
        if internal_shuffle_flag: #if shuffle -- get whatever glob return
            image_files = dict_of_lists['list_of_all_files']
            all_labels = dict_of_lists['labels_from_file']
        else: #if not shuffle -- order by sample ID
            image_files = dict_of_lists['filenames']
            all_labels = dict_of_lists['labels']
        # get class label for one task only
        one_task_label = get_label_one_task_only(all_labels, task_name)
        return_dict = transform_class_labels_to_number(one_task_label)
        labels_by_class_numbers = return_dict['all_class_number']
        num_classes = return_dict['num_classes']
        all_classes = return_dict['all_classes']
        map_label_to_num = return_dict['map_label_to_num']
        #Return 
        dict_of_objects = {
            'path_to_folder':path_to_folder,
            'image_files':image_files,
            'one_task_label':one_task_label,
            'labels_by_class_numbers':labels_by_class_numbers,
            'total_samples':total_samples
        }
        return dict_of_objects

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        img_path = os.path.join(self.path_to_folder, self.image_files[idx])
        image = read_image(img_path)
        label = self.labels_by_class_numbers[idx]
        if self.transform:
            image = self.transform(image)
        return image, label            



########################################################################## 
# For the simple set (SIMPLE4MD)
########################################################################## 

class HEPT4MD_CustomizeDataset_SIMPLE(Dataset):
    def __init__(self, dict_of_exp_info, dataset_seed =12345, transform=None, internal_shuffle_flag = False):
        self.dataset_seed = dataset_seed
        self.internal_shuffle_flag = internal_shuffle_flag
        self.task_name = dict_of_exp_info['task_name']  #'SD', 'LR', 'AB'
        self.obj_config = dict_of_exp_info['obj_config'] # 'BOTH' , 'ID', 'LOC'
        self.num_distractor = dict_of_exp_info['num_distractor'] # 0, 1, 2, 3
        self.set_type = dict_of_exp_info['set_type'] #train, test, validation
        
        #Get paths and labels
        dict_of_objects = self.get_all_image_paths_and_labels(dict_of_exp_info, internal_shuffle_flag)
        self.path_to_folder = dict_of_objects['path_to_folder']
        self.image_files = dict_of_objects['image_files']
        self.one_task_label = dict_of_objects['one_task_label']
        self.labels_by_class_numbers = dict_of_objects['labels_by_class_numbers']
        self.total_samples = dict_of_objects['total_samples']

        self.transform = transform

    def get_all_image_paths_and_labels(self, dict_of_exp_info, internal_shuffle_flag):

        from info_scripts.info_dataset import return_get_path_to_folder_function, list_of_files_in_data_folder, get_label_one_task_only, transform_class_labels_to_number
        
        get_path_to_folder = return_get_path_to_folder_function(RUNNING_MODE = 'SIMPLE')
        dataset_seed = self.dataset_seed
        task_name = dict_of_exp_info['task_name'] #'SD', 'LR', 'AB'
        obj_config = dict_of_exp_info['obj_config']
        num_distractor = dict_of_exp_info['num_distractor']
        set_type = dict_of_exp_info['set_type'] #train, test, validation
        path_to_folder = get_path_to_folder(dataset_seed, task_name, obj_config, num_distractor, set_type)
        dict_of_lists = list_of_files_in_data_folder(path_to_folder)
        total_samples = dict_of_lists['total_samples']
        if internal_shuffle_flag: #if shuffle -- get whatever glob return
            image_files = dict_of_lists['list_of_all_files']
            all_labels = dict_of_lists['labels_from_file']
        else: #if not shuffle -- order by sample ID
            image_files = dict_of_lists['filenames']
            all_labels = dict_of_lists['labels']
        # get class label for one task only
        one_task_label = get_label_one_task_only(all_labels, task_name)
        return_dict = transform_class_labels_to_number(one_task_label)
        labels_by_class_numbers = return_dict['all_class_number']
        num_classes = return_dict['num_classes']
        all_classes = return_dict['all_classes']
        map_label_to_num = return_dict['map_label_to_num']
        #Return 
        dict_of_objects = {
            'path_to_folder':path_to_folder,
            'image_files':image_files,
            'one_task_label':one_task_label,
            'labels_by_class_numbers':labels_by_class_numbers,
            'total_samples':total_samples
        }
        return dict_of_objects

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        img_path = os.path.join(self.path_to_folder, self.image_files[idx])
        image = read_image(img_path)
        label = self.labels_by_class_numbers[idx]
        if self.transform:
            image = self.transform(image)
        return image, label        

########################################################################## 
# Most Generic Version  ---> set set_type = alphabet set directly
# -- can specify the folder path seed number
########################################################################## 

class HEPT4MD_CustomizeDataset_byALPHABET(Dataset):
    def __init__(self, dict_of_exp_info, dataset_seed =12345, transform=None, internal_shuffle_flag = False):
        self.dataset_seed = dataset_seed
        self.internal_shuffle_flag = internal_shuffle_flag
        self.task_name = dict_of_exp_info['task_name']  #'SD', 'LR', 'AB'
        self.obj_config = dict_of_exp_info['obj_config'] # 'SA' , 'ID', 'LOC'
        self.num_distractor = dict_of_exp_info['num_distractor'] # 0, 1, 2, 3
        self.set_type = dict_of_exp_info['set_type'] #train, test, validation
        
        #Get paths and labels
        dict_of_objects = self.get_all_image_paths_and_labels(dict_of_exp_info, internal_shuffle_flag)
        self.path_to_folder = dict_of_objects['path_to_folder']
        self.image_files = dict_of_objects['image_files']
        self.one_task_label = dict_of_objects['one_task_label']
        self.labels_by_class_numbers = dict_of_objects['labels_by_class_numbers']
        self.total_samples = dict_of_objects['total_samples']

        self.transform = transform

    def get_all_image_paths_and_labels(self, dict_of_exp_info, internal_shuffle_flag):

        from info_scripts.info_dataset import return_get_path_to_folder_function, list_of_files_in_data_folder, get_label_one_task_only, transform_class_labels_to_number
        
        get_path_to_folder = return_get_path_to_folder_function(RUNNING_MODE = 'ALPHABET')
        dataset_seed = self.dataset_seed
        task_name = dict_of_exp_info['task_name'] #'SD', 'LR', 'AB'
        obj_config = dict_of_exp_info['obj_config']
        num_distractor = dict_of_exp_info['num_distractor']
        set_type = dict_of_exp_info['set_type'] #train, test, validation
        path_to_folder = get_path_to_folder(dataset_seed, task_name, obj_config, num_distractor, set_type)
        dict_of_lists = list_of_files_in_data_folder(path_to_folder)
        total_samples = dict_of_lists['total_samples']
        if internal_shuffle_flag: #if shuffle -- get whatever glob return
            image_files = dict_of_lists['list_of_all_files']
            all_labels = dict_of_lists['labels_from_file']
        else: #if not shuffle -- order by sample ID
            image_files = dict_of_lists['filenames']
            all_labels = dict_of_lists['labels']
        # get class label for one task only
        one_task_label = get_label_one_task_only(all_labels, task_name)
        return_dict = transform_class_labels_to_number(one_task_label)
        labels_by_class_numbers = return_dict['all_class_number']
        num_classes = return_dict['num_classes']
        all_classes = return_dict['all_classes']
        map_label_to_num = return_dict['map_label_to_num']
        #Return 
        dict_of_objects = {
            'path_to_folder':path_to_folder,
            'image_files':image_files,
            'one_task_label':one_task_label,
            'labels_by_class_numbers':labels_by_class_numbers,
            'total_samples':total_samples
        }
        return dict_of_objects

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        img_path = os.path.join(self.path_to_folder, self.image_files[idx])
        image = read_image(img_path)
        label = self.labels_by_class_numbers[idx]
        if self.transform:
            image = self.transform(image)
        return image, label        
