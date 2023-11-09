"""
	DATASET INFORMATION

	#exec(open("info_dataset.py").read()) 
"""

"""
######################################################################################################################
### HEADING
######################################################################################################################

#===============================================================================
# SECTION #
#===============================================================================

#---------------------------------------------------------------
### sub-section#
#---------------------------------------------------------------
"""
#SIMPLE4MD
PATH_TO_SIMPLE_SET_FOLDER = "/media/data_cifs/projects/prj_visreason/dcsr/hept_for_human/SIMPLE4MD/"
PATH_TO_SIMPLE_SET_FOLDER_SEED12345 = PATH_TO_SIMPLE_SET_FOLDER + "For_NumDTT_EXP_SDxLRxAB_S12345_combinePNG/"
PATH_TO_SIMPLE_SET_FOLDER_SEED54321 = PATH_TO_SIMPLE_SET_FOLDER + "For_NumDTT_EXP_SDxLRxAB_S54321_NewShapes_combinePNG/"

# Generic Ver
PATH_TO_DATASET_FOLDER =  "/media/data_cifs/projects/prj_visreason/dcsr/hept_for_human/MD_EXP/NumberOfDistractors_SDxLRxAB_CombinePNG_allConfigs/"

PATH_TO_DATASET_FOLDER_SEED12345 = "/media/data_cifs/projects/prj_visreason/dcsr/hept_for_human/MD_EXP/NumberOfDistractors_SDxLRxAB_Seed12345_combinePNG/"

PATH_TO_DATASET_FOLDER_SEED778899 = "/media/data_cifs/projects/prj_visreason/dcsr/hept_for_human/MD_EXP_TEST_SET/NumberOfDistractors_SDxLRxAB_Seed778899_CombinePNG/"

PATH_TO_DATASET_FOLDER_DECOMINOES = "/media/data_cifs/projects/prj_visreason/dcsr/hept_for_human/MD_EXP_POLYOMINO/DECOMINOES_SDxLRxAB_SEED778899_combinePNG/"
PATH_TO_DATASET_FOLDER_TETROMINOES = "/media/data_cifs/projects/prj_visreason/dcsr/hept_for_human/MD_EXP_POLYOMINO/TETROMINOES_SDxLRxAB_SEED778899_combinePNG/"


DATASET_SUBSET = {'A':'SetA/', 'B':'SetB/', 'C':'SetC/', 'D':'SetD/' }

DATAFOLDER_BY_EXP = ['S5M8D0/', 'S5M8D1/', 'S5M8D2/', 'S5M8D3/']

#############################################################################################################################
### Information about the folder organization
#############################################################################################################################

"""
FOLDER ORGANIZATION:
	SetX/ 
		> AB-ObjID_NonTrivialCase/
		> AB-ObjLOC_TrivialCase/
		> AB-ObjSA_BasicCase/
		> AB-ObjPMT_SpecificCase/
		> SD-ObjID_TrivialCase/
		> SD-ObjLOC_NonTrivialCase/
		> SD-ObjPMT_SpecificCase/
		> SD-ObjSA_BasicCase/
			> S5M8D0/
			> S5M8D1/
			> S5M8D2/
			> S5M8D3/
				> OrgCanvas/
				> AoI/
								> sample#_nd#_{CONFIG}_{TASK}_{XXX:labels}.png
									e.g.,  sample9_nd1_ID_AB_DLB.png
"""

DATA_FOLDERS_BY_TASK_CONFIG_PAIR = { 
#Task: Above-Below
('AB', 'ID'): 'AB-ObjID_NonTrivialCase/',
('AB', 'LOC'): 'AB-ObjLOC_TrivialCase/',
('AB', 'SA'): 'AB-ObjSA_BasicCase/',
('AB', 'PMT'): 'AB-ObjPMT_SpecificCase/',
#Task: Same-Different
('SD', 'ID'): 'SD-ObjID_TrivialCase/',
('SD', 'LOC'): 'SD-ObjLOC_NonTrivialCase/',
('SD', 'SA'): 'SD-ObjSA_BasicCase/',
('SD', 'PMT'): 'SD-ObjPMT_SpecificCase/',
}

DATA_FOLDERS_BY_TASK_CONFIG_PAIR_NEW_VER = { 
#Task: Above-Below
('AB', 'ID'): 'AB-ObjID_NonTrivialCase/',
('AB', 'LOC'): 'AB-ObjLOC_TrivialCase/',
#('AB', 'SA'): 'AB-ObjSA_BasicCase/',
('AB', 'SA'): 'AB-ObjBOTH_BasicCase/',
('AB', 'PMT'): 'AB-ObjPMT_SpecificCase/',
#Task: Same-Different
('SD', 'ID'): 'SD-ObjID_TrivialCase/',
('SD', 'LOC'): 'SD-ObjLOC_NonTrivialCase/',
('SD', 'SA'): 'SD-ObjBOTH_BasicCase/',
('SD', 'PMT'): 'SD-ObjPMT_SpecificCase/',
}

DATA_FOLDERS_BY_TASK_CONFIG_PAIR_SIMPLE_SET = {
#Task: Above-Below
('AB', 'ID'): 'AB-ObjID_NonTrivialCase/',
('AB', 'LOC'): 'AB-ObjLOC_TrivialCase/',
#('AB', 'SA'): 'AB-ObjSA_BasicCase/',
('AB', 'BOTH'): 'AB-ObjBOTH_BasicCase/',
('AB', 'PMT'): 'AB-ObjPMT_SpecificCase/',
#Task: Same-Different
('SD', 'ID'): 'SD-ObjID_TrivialCase/',
('SD', 'LOC'): 'SD-ObjLOC_NonTrivialCase/',
('SD', 'BOTH'): 'SD-ObjBOTH_BasicCase/',
('SD', 'PMT'): 'SD-ObjPMT_SpecificCase/',
}
#############################################################################################################################
### LITE-VERSION 
### ---> trainSet, testSet, valSet = 'A', 'B', 'C'
#############################################################################################################################

SET_NAME_LITE_VER = {'train': 'A', 'test':'B', 'validation':'C'}  # ---> the name was wrong T^T During HP search, I validate my model performance on set B
SET_NAME_GENERAL_VER = {'train': 'A', 'development':'B', 'validation':'C', 'final_evaluation':'D'}
SET_NAME_SIMPLE_VER = {'train': 'A', 'development':'B', 'validation':'C', 'train_more':'D'}


def get_path_to_folder_lite_ver(task_name, obj_config, num_distractor, set_type):
	return PATH_TO_DATASET_FOLDER+DATASET_SUBSET[SET_NAME_LITE_VER[set_type]] + DATA_FOLDERS_BY_TASK_CONFIG_PAIR[(task_name, obj_config)] + DATAFOLDER_BY_EXP[num_distractor] + 'OrgCanvas/'

def get_path_to_folder_general_ver(task_name, obj_config, num_distractor, set_type):
	return PATH_TO_DATASET_FOLDER+DATASET_SUBSET[SET_NAME_GENERAL_VER[set_type]] + DATA_FOLDERS_BY_TASK_CONFIG_PAIR[(task_name, obj_config)] + DATAFOLDER_BY_EXP[num_distractor] + 'OrgCanvas/'

def get_path_to_folder_with_seed_ver(dataset_seed, task_name, obj_config, num_distractor, set_type):
	if dataset_seed == 12345:
		path_to_data_folder = PATH_TO_DATASET_FOLDER_SEED12345
		path_to_task_config_folder = DATA_FOLDERS_BY_TASK_CONFIG_PAIR
	elif dataset_seed == 778899:
		path_to_data_folder = PATH_TO_DATASET_FOLDER_SEED778899
		path_to_task_config_folder = DATA_FOLDERS_BY_TASK_CONFIG_PAIR_NEW_VER
	else:
		path_to_data_folder = PATH_TO_DATASET_FOLDER
		path_to_task_config_folder = DATA_FOLDERS_BY_TASK_CONFIG_PAIR

	return path_to_data_folder+DATASET_SUBSET[SET_NAME_GENERAL_VER[set_type]] + path_to_task_config_folder[(task_name, obj_config)] + DATAFOLDER_BY_EXP[num_distractor] + 'OrgCanvas/'


def get_path_to_folder_polyominoes(task_name, obj_config, num_distractor, set_type, poly_set='HEPT', dataset_seed = 778899):
	if poly_set == 'DECO':
		path_to_data_folder = PATH_TO_DATASET_FOLDER_DECOMINOES
		path_to_task_config_folder = DATA_FOLDERS_BY_TASK_CONFIG_PAIR_NEW_VER
	elif poly_set == 'TETRO':
		path_to_data_folder = PATH_TO_DATASET_FOLDER_TETROMINOES
		path_to_task_config_folder = DATA_FOLDERS_BY_TASK_CONFIG_PAIR_NEW_VER
	else: #HEPTO
		path_to_data_folder = PATH_TO_DATASET_FOLDER_SEED778899
		path_to_task_config_folder = DATA_FOLDERS_BY_TASK_CONFIG_PAIR_NEW_VER

	return path_to_data_folder+DATASET_SUBSET[SET_NAME_GENERAL_VER[set_type]] + path_to_task_config_folder[(task_name, obj_config)] + DATAFOLDER_BY_EXP[num_distractor] + 'OrgCanvas/'


def get_path_to_folder_with_SIMPLE4MD(dataset_seed, task_name, obj_config, num_distractor, set_type):
	if dataset_seed == 12345:
		path_to_data_folder = PATH_TO_SIMPLE_SET_FOLDER_SEED12345
		path_to_task_config_folder = DATA_FOLDERS_BY_TASK_CONFIG_PAIR_SIMPLE_SET
	elif dataset_seed == 54321:
		path_to_data_folder = PATH_TO_SIMPLE_SET_FOLDER_SEED54321
		path_to_task_config_folder = DATA_FOLDERS_BY_TASK_CONFIG_PAIR_SIMPLE_SET
	else:
		path_to_data_folder = PATH_TO_SIMPLE_SET_FOLDER_SEED12345
		path_to_task_config_folder = DATA_FOLDERS_BY_TASK_CONFIG_PAIR_SIMPLE_SET

	return path_to_data_folder+DATASET_SUBSET[SET_NAME_SIMPLE_VER[set_type]] + path_to_task_config_folder[(task_name, obj_config)] + DATAFOLDER_BY_EXP[num_distractor] + 'OrgCanvas/'


def get_path_to_folder_with_seed_ver_alphabet(dataset_seed, task_name, obj_config, num_distractor, set_type):
	#set_type = {'A', 'B', 'C', 'D'}
	if dataset_seed == 12345:
		path_to_data_folder = PATH_TO_DATASET_FOLDER_SEED12345
		path_to_task_config_folder = DATA_FOLDERS_BY_TASK_CONFIG_PAIR
	elif dataset_seed == 778899:
		path_to_data_folder = PATH_TO_DATASET_FOLDER_SEED778899
		path_to_task_config_folder = DATA_FOLDERS_BY_TASK_CONFIG_PAIR_NEW_VER
	elif dataset_seed == 10: #POLYOMINOES "DECO" Set
		path_to_data_folder = PATH_TO_DATASET_FOLDER_DECOMINOES
		path_to_task_config_folder = DATA_FOLDERS_BY_TASK_CONFIG_PAIR_NEW_VER
	elif dataset_seed == 4: #POLYOMINOES "TETRO" Set
		path_to_data_folder = PATH_TO_DATASET_FOLDER_TETROMINOES
		path_to_task_config_folder = DATA_FOLDERS_BY_TASK_CONFIG_PAIR_NEW_VER			
	else:
		path_to_data_folder = PATH_TO_DATASET_FOLDER
		path_to_task_config_folder = DATA_FOLDERS_BY_TASK_CONFIG_PAIR

	return path_to_data_folder+DATASET_SUBSET[set_type] + path_to_task_config_folder[(task_name, obj_config)] + DATAFOLDER_BY_EXP[num_distractor] + 'OrgCanvas/'


#############################################################################################################################
### Other Helper Functions
#############################################################################################################################
RUNNING_MODE = 'HPsearch' #'GENERAL' 'SEED'
def return_get_path_to_folder_function(RUNNING_MODE = 'GENERAL'):
	if RUNNING_MODE == 'GENERAL':
		#For Real Experiment
		get_path_to_folder = get_path_to_folder_general_ver
	elif RUNNING_MODE == 'HPsearch':
		#For HP Search
		get_path_to_folder = get_path_to_folder_lite_ver
	elif RUNNING_MODE == 'SEED':
		get_path_to_folder = get_path_to_folder_with_seed_ver
	elif RUNNING_MODE == 'POLY':
		get_path_to_folder = get_path_to_folder_polyominoes
	elif RUNNING_MODE == 'SIMPLE':
		get_path_to_folder = get_path_to_folder_with_SIMPLE4MD
	elif RUNNING_MODE == 'ALPHABET': #HEPT4MD | call set by alphabet
		get_path_to_folder = get_path_to_folder_with_seed_ver_alphabet		
	else: #default
		get_path_to_folder = get_path_to_folder_with_seed_ver
	return get_path_to_folder

get_path_to_folder = return_get_path_to_folder_function(RUNNING_MODE)



"""
def list_of_files_in_data_folder(path_to_folder):
	import glob, os
	#current_dir = os.getcwd()
	list_of_all_files = []
	labels_from_file = []
	
	#os.chdir(path_to_folder)
	names_all_files = [os.path.basename(x) for x in glob.glob(path_to_folder+'*.png')]
	for file in names_all_files:
		list_of_all_files.append(file)		
		#example 'sample2954_nd1_SA_AB_DRA.png'
		lbl_png = file.split('_')[-1]
		lbl = lbl_png.split('.')[0]
		labels_from_file.append(lbl)
	total_samples = len(list_of_all_files)
	filenames_in_order = [None]*total_samples
	label_in_order = [None]*total_samples
	for f_i in range(total_samples):
		this_file = list_of_all_files[f_i]
		this_label = labels_from_file[f_i]
		this_sample = this_file.split('_')[0]
		sample_id = int(this_sample[6:])
		filenames_in_order[sample_id] = this_file
		label_in_order[sample_id] = this_label
	dict_of_lists = {
	'list_of_all_files':list_of_all_files,
	'labels_from_file':labels_from_file,
	'filenames':filenames_in_order,
	'labels':label_in_order,
	'total_samples':total_samples
	}
	os.chdir(current_dir)
	return dict_of_lists
"""

def list_of_files_in_data_folder(path_to_folder):
	import glob, os
	list_of_all_files = []
	labels_from_file = []
	names_all_files = [os.path.basename(x) for x in glob.glob(path_to_folder+'*.png')]
	for file in names_all_files:
		list_of_all_files.append(file)		
		#example 'sample2954_nd1_SA_AB_DRA.png'
		lbl_png = file.split('_')[-1]
		lbl = lbl_png.split('.')[0]
		labels_from_file.append(lbl)
	total_samples = len(list_of_all_files)
	filenames_in_order = [None]*total_samples
	label_in_order = [None]*total_samples
	for f_i in range(total_samples):
		this_file = list_of_all_files[f_i]
		this_label = labels_from_file[f_i]
		this_sample = this_file.split('_')[0]
		sample_id = int(this_sample[6:])
		filenames_in_order[sample_id] = this_file
		label_in_order[sample_id] = this_label
	dict_of_lists = {
	'list_of_all_files':list_of_all_files,
	'labels_from_file':labels_from_file,
	'filenames':filenames_in_order,
	'labels':label_in_order,
	'total_samples':total_samples
	}
	return dict_of_lists



def get_label_one_task_only(all_labels, task_name):
	one_task_label = []
	for lbl_3tasks in all_labels:
		if task_name == 'SD':
			lbl = lbl_3tasks[0]
		elif task_name == 'LR':
			lbl = lbl_3tasks[1]
		elif task_name == 'AB':
			lbl = lbl_3tasks[2]
		else:
			lbl = lbl_3tasks
		one_task_label.append(lbl)
	return one_task_label

def transform_class_labels_to_number(all_labels):
	import numpy as np
	all_classes = np.unique(all_labels)
	num_classes = len(all_classes)
	map_label_to_num = {}
	for c_i in range(num_classes):
		map_label_to_num[all_classes[c_i]] = c_i
	list_of_class_number = []
	for lbl in all_labels:	
		list_of_class_number.append(map_label_to_num[lbl])
	return_dict = {
	'all_class_number':list_of_class_number,
	'num_classes':num_classes,
	'all_classes':list(all_classes),
	'map_label_to_num':map_label_to_num
	}
	return return_dict

"""
	Example of the processing pipeline
"""
def get_all_image_paths_and_labels(dict_of_exp_info, shuffle_flag):
	#from info_scripts.info_dataset import get_path_to_folder, list_of_files_in_data_folder, get_label_one_task_only, transform_class_labels_to_number
    task_name = dict_of_exp_info['task_name'] #'SD', 'LR', 'AB'
    obj_config = dict_of_exp_info['obj_config']
    num_distractor = dict_of_exp_info['num_distractor']
    set_type = dict_of_exp_info['set_type'] #train, test, validation
    path_to_folder = get_path_to_folder(task_name, obj_config, num_distractor, set_type)
    dict_of_lists = list_of_files_in_data_folder(path_to_folder)
    total_samples = dict_of_lists['total_samples']
    if shuffle_flag: #if shuffle -- get whatever glob return
        image_files = dict_of_lists['list_of_all_files']
        all_labels = dict_of_lists['labels_from_file']
    else: #if not shuffle -- order by sample ID
        image_files = dict_of_lists['filenames']
        all_labels = dict_of_lists['labels']
    # get class label for one task only
    one_task_label = get_label_one_task_only(all_labels, task_name)
    return_dict = transform_class_labels_to_number(labels)
    labels_by_class_numbers = return_dict['all_class_number']
    num_classes = return_dict['num_classes']
    all_classes = return_dict['all_classes']
    map_label_to_num = return_dict['map_label_to_num']
    #Return 
    dict_of_objects = {
        'path_to_folder':path_to_folder,
        'image_files':image_files,
        'one_task_label':one_task_label,
        'labels_by_class_numbers':labels_by_class_numbers
    }
    return dict_of_objects


#############################################################################################################################




#############################################################################################################################
#############################################################################################################################
#####  The "LATER" FULL Experiment
#############################################################################################################################
#############################################################################################################################
### Information for all FOUR Experimental Trials
### > For each trial
### 		> 3 sets for hyper-parameter search and training set
###			> 1 set for validation set (=testing set for each experimental trial)
#############################################################################################################################

template_for_experimental_trial = { 'HPsearch_and_TRAINING_SET':[], 'VALIDATION_SET':[''] }
#===============================================================================
## TRIAL 1 
#===============================================================================
TRIAL1_SETINFO = { 'HPsearch_and_TRAINING_SET':['A', 'B', 'C'], 'VALIDATION_SET':['D'] }

#===============================================================================
## TRIAL 2
#===============================================================================
TRIAL2_SETINFO = { 'HPsearch_and_TRAINING_SET':['B', 'D', 'A'], 'VALIDATION_SET':['C'] }

#===============================================================================
## TRIAL 3
#===============================================================================
TRIAL3_SETINFO = { 'HPsearch_and_TRAINING_SET':['C', 'A', 'D'], 'VALIDATION_SET':['B'] }

#===============================================================================
## TRIAL 4
#===============================================================================
TRIAL4_SETINFO = { 'HPsearch_and_TRAINING_SET':['D', 'C', 'B'], 'VALIDATION_SET':['A'] }


#############################################################################################################################
### For each experimental trial
### 	> Hyper-parameter search: 2 attempts (= search trial)
###		> training: 3 attempts (= training trial)
#############################################################################################################################