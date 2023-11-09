"""
	GAMR-tripleZ (22175 MiB) Model HP search on HEPT4MD-MORE dataset   ---> Train on 'A',  Test on 'C'
        Edited from SIMPLE4MD-more_GAMR-tripleZ_HPsearch.py
        Last Edited: 10/23/2023
    #exec(open("HEPT4MD-AC_GAMR-TripleZ_HPsearch.py").read())
    WandB: https://wandb.ai/p-sailamul/HEPT4AC_GAMR-tripleZ_HP
"""

"""
#Test that the code work  
    First ---> with local_test_mode = True, then use bash script

#on Z8 - session1

#Trivial Cases
CUDA_VISIBLE=0, python HEPT4MD-AC_GAMR-TripleZ_HPsearch.py -t AB -o LOC -d 1 -H CONST -L 1E-4 -w 1E-4 -T 4 -R 999999 -p 5 -b 60
CUDA_VISIBLE=0, python HEPT4MD-AC_GAMR-TripleZ_HPsearch.py -t SD -o ID -d 1 -H CONST -L 1E-4 -w 1E-4 -T 4 -R 999999 -p 5 -b 60

#Non-Trivial Cases
CUDA_VISIBLE=0, python HEPT4MD-AC_GAMR-TripleZ_HPsearch.py -t AB -o ID -d 1 -H CONST -L 1E-4 -w 1E-4 -T 4 -R 999999 -p 50 -b 60
CUDA_VISIBLE=0, python HEPT4MD-AC_GAMR-TripleZ_HPsearch.py -t SD -o LOC -d 1 -H CONST -L 1E-4 -w 1E-4 -T 4 -R 999999 -p 50 -b 60

"""

"""
Best of SD/SA/ND1 == 0.9 ---> LR=9E-4, WD=0, T=4
#SD/LOC/ND1 --- Z7 - session4
CUDA_VISIBLE=0, python HEPT4MD-AC_GAMR-TripleZ_HPsearch.py -t SD -o LOC -d 1 -H CONST -L 9E-4 -w 0 -T 4 -R 999999 -p 50 -b 60
#SD/ID/ND1  --- Z5 - session4
CUDA_VISIBLE=0, python HEPT4MD-AC_GAMR-TripleZ_HPsearch.py -t SD -o ID -d 1 -H CONST -L 9E-4 -w 0 -T 4 -R 999999 -p 50 -b 60
"""

"""
Best of AB/ID/ND1 == 0.7195 ---> LR=1E-4, WD=0.01, T=4
#AB/LOC/ND1 --- Z2 - session4
CUDA_VISIBLE=0, python HEPT4MD-AC_GAMR-TripleZ_HPsearch.py -t AB -o LOC -d 1 -H CONST -L 1E-4 -w 0.01 -T 4 -R 999999 -p 50 -b 60
CUDA_VISIBLE=0, python HEPT4MD-AC_GAMR-TripleZ_HPsearch.py -t AB -o SA -d 1 -H CONST -L 1E-4 -w 0.01 -T 4 -R 999999 -p 50 -b 60
"""

"""
echo "< Z7 > [START] HEPT4MD-AC - GAMR-TripleZ - HP search"
echo "SEED = 999999"
echo "SD/ID/ND1"
#Z2 - session4
CUDA_VISIBLE=0, python HEPT4MD-AC_GAMR-TripleZ_HPsearch.py -t SD -o ID -d 1 -H CONST -L 1E-5 -w 0 -T 4 -R 999999 -p 50 -b 60
#Z5 - session4
CUDA_VISIBLE=0, python HEPT4MD-AC_GAMR-TripleZ_HPsearch.py -t SD -o ID -d 1 -H CONST -L 1E-3 -w 0 -T 4 -R 999999 -p 50 -b 60
"""

"""
SD/ID/ND1 ---> LR:1E-4, WD:1E-4, T:4  --- try multiple seeds
CUDA_VISIBLE=0, python HEPT4MD-AC_GAMR-TripleZ_HPsearch.py -t SD -o ID -d 1 -H CONST -L 1E-4 -w 1E-4 -T 4 -R 888888 -p 50 -b 60
CUDA_VISIBLE=0, python HEPT4MD-AC_GAMR-TripleZ_HPsearch.py -t SD -o ID -d 1 -H CONST -L 1E-4 -w 1E-4 -T 4 -R 777777 -p 50 -b 60
CUDA_VISIBLE=0, python HEPT4MD-AC_GAMR-TripleZ_HPsearch.py -t SD -o ID -d 1 -H CONST -L 1E-4 -w 1E-4 -T 4 -R 778899 -p 50 -b 60
CUDA_VISIBLE=0, python HEPT4MD-AC_GAMR-TripleZ_HPsearch.py -t SD -o ID -d 1 -H CONST -L 1E-4 -w 1E-4 -T 4 -R 998877 -p 50 -b 60
"""

######################################################################################################################
### SET-UP PYTHON
######################################################################################################################
import sys
sys.path.append("/media/data_cifs/projects/prj_visreason/dcsr/common_scripts/")
#import utils
#Or
from utils import *

######################################################################################################################
### IMPORT
######################################################################################################################
import numpy as np
import random
import json

import os
import copy
import signal
import argparse
from tqdm import tqdm

#-------------------------------------------
# Torch related
#-------------------------------------------
#torch related
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

#-------------------------------------------
# import custom classes 
#-------------------------------------------
from ImageDataset import HEPT4MD_CustomizeDataset_byALPHABET
from model import resnet_unify
#import models.GAMR as GAMR
from GAMR import GAMR, GAMR_tripleZ
from baseline.models.modules import *

#-------------------------------------------
# Other Scripts
#-------------------------------------------
import info_scripts.info_dataset as info_dataset
 
######################################################################################################################
### PREPARATION: DEFAULT PARAMETERS
######################################################################################################################

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD =[0.229, 0.224, 0.225]
DEFAULT_RANDOM_SEED = 778899
#-------------------------------------------
# DATASET information
#-------------------------------------------
DEFAULT_MORE_DATA_FLAG = False
DATASET_NAME = 'HEPT4MD-AC'
INTERNAL_SHUFFLE_FLAG = False
#-------------------------------------------
# Defaults Model and Training Set-Up
#-------------------------------------------

DEFAULT_GRAD_CLIP_VAL = 1.0
DEFAULT_FROZEN_FLAG = False #For Resnet 50
DEFAULT_MAX_EPOCH = 50
DEFAULT_BATCH_SIZE = 60
DEFAULT_CRITERION_CODE = 'CETP'
DEFAULT_OPTIMIZER_CODE = 'ADAMW'
DEFAULT_WEIGHT_DECAY = 0
DEFAULT_SCHEDULER_CODE = 'CONST'
DEFAULT_LR = 1e-4
DEFAULT_LR_M = 0.1
DEFAULT_LR_S = 4

DEFAULT_NUM_VAL_SAMPLE = 1920
DEFAULT_EARLY_STOP_VAL_ACC = 0.9999
EARLY_STOP_VAL_ACC = DEFAULT_EARLY_STOP_VAL_ACC
CEILING_OF_TRAIN_ACC = 0.9999
CONDITION_VAL_ACC_OF_CEILING_TRAIN = 0.55
DEFAULT_SHUFFLE_FLAG = True

DEFAULT_GAMR_TIME_STEP = 4
MODEL_NAME = 'GAMR-tripleZ' #Resnet50
DEFAULT_MODEL_CODE = 3
# 0: use the default encoder (Enc_Conv_v0_16)
# 1: use Resnet50 without pretrain
# 2: use Resnet50 WITH pretrain
# 3: GAMR-tripleZ = use 3 channels of Conv --> Enc_Conv_splitCH
# 4: GAMR-Psych = Use Conv Encoder with Z_img ratio Tar:Ref:Full = 32:32:64

CNT_NUM_CEILING_BEFORE_STOP = 10

#-------------------------------------------
# For configuration
#-------------------------------------------

BASE_RUN_CODE = DATASET_NAME
SUFFIX_RUN_CODE = 'HP'

RECORD_HOST_NAME_FLAG = True
######################################################################################################################
### PREPARATION: HYPER-PARAMETERS |****************************************************************************************
######################################################################################################################

LOCAL_TEST_MODE = False #True #False
RUN_WandB_as_scratch = False #True #False
AUTORUN = True
RUN_SWEEP = False 

PRESET_THIS_TASK = 'AB' #'SD'
PRESET_THIS_CONFIG = 'ID' #'ID' #'LOC'
PRESET_THIS_ND = 1
PRESET_THIS_SCHEDULER = DEFAULT_SCHEDULER_CODE #CONST, 'StepLR', 'AnnealingLR' #Note: 'CyclicLR' doesn't work with ADAM
PRESET_THIS_LR = DEFAULT_LR #1e-2, 1e-3, 1e-4, 1e-5
PRESET_GAMR_TIME_STEP = DEFAULT_GAMR_TIME_STEP #
########################################################
# SET SEED
########################################################
set_all_random_seed(input_seed=DEFAULT_RANDOM_SEED)
set_all_torch_random_seed(seed=DEFAULT_RANDOM_SEED)   

########################################################
# SETUP Weights and Biases
########################################################
import wandb

DANGER_API_KEY = '76412af466364361f810728b4c9fd302904173c6'
os.environ["WANDB_API_KEY"] = DANGER_API_KEY

if RUN_WandB_as_scratch:
    WANDB_PROJECT= "scratch"
else:     
    WANDB_PROJECT= "HEPT4AC_GAMR-tripleZ_HP"
    #WANDB_PROJECT= "HEPT4AC_GAMR-Psych_HP"

WANDB_LOG_ALL_FLAG = False # For HP-search = False


########################################################
# PATHS   --- this should be read from config file    
########################################################
#-------------------------------------------------------------------------------------------------------
# GENERAL
#-------------------------------------------------------------------------------------------------------
current_path = os.getcwd() + '/'

PATH_TO_SAVE_MODEL_STATE = "all_models_state_dicts/HEPT4MD-AC_GAMR-tripleZ_HPsearch/"
create_folder_if_not_exists(path_to_folder=current_path+PATH_TO_SAVE_MODEL_STATE, desc="Folder to save model's state_dict()")

PATH_TO_SAVE_LOG_FILE = "all_log_runs_info/HEPT4MD-AC_GAMR-tripleZ_HPsearch/"
create_folder_if_not_exists(path_to_folder=current_path+PATH_TO_SAVE_LOG_FILE, desc="Folder to save internal log (information about this run)")


########################################################
# The Architecture
########################################################
#from model import resnet_unify

######################################################################################################################
### LOCAL FUNCTIONS
######################################################################################################################

def generate_running_code_from_config_dict(config_dict, base_run_code=BASE_RUN_CODE, suffix_run_code = SUFFIX_RUN_CODE):
    all_keys = config_dict.keys()
    running_code = f"{base_run_code}"
    for key in all_keys:
        this_val = config_dict[key]
        if not isinstance(this_val, float) :
            running_code += f"__{key}_{this_val}"
        else:
            running_code += f"__{key}_{this_val:.1E}"
    if suffix_run_code != '':
        running_code += f"__{suffix_run_code}"
    return running_code


# -----------------------------------------------------------
## Selecting Models
# -----------------------------------------------------------
def return_resnet_model_name_from_code(model_code, frozen_features_flag, baseRN_type=50):
    this_model_name = f"ResNet{baseRN_type}-{model_code}"
    if frozen_features_flag:
        this_model_name = f"{this_model_name}-frzn"
    else:
        this_model_name = f"{this_model_name}-unfz"
    return this_model_name
    
def return_RN_model_by_code(model_code = 0, baseRN_type=50, frozen_features_flag = True, return_name_flag=False, *args, **kwargs):
    #from models import resnet_unify
    #Defaults
    bool_keep_avgpool=True
    bool_add_bufferFC=False
    locals().update(kwargs) #bool(kwargs) = False is empty
    if model_code == 1: #keep avgpoool, with buffer
        bool_keep_avgpool = True
        bool_add_bufferFC = True
    elif model_code == 2:#remove avgpool, with buffer by default
        bool_keep_avgpool = False
        bool_add_bufferFC = True
    else: #Default
        bool_keep_avgpool = bool_keep_avgpool
        bool_add_bufferFC = bool_add_bufferFC
    model = resnet_unify(baseRN = baseRN_type, frozen_features_flag = frozen_features_flag, keep_pool_flag=bool_keep_avgpool, buffer_fc_flag = bool_add_bufferFC, *args,**kwargs)
    
    this_model_name = f"ResNet{baseRN_type}-{model_code}"
    if frozen_features_flag:
        this_model_name = f"{this_model_name}-frzn"
    else:
        this_model_name = f"{this_model_name}-unfz"
        
    if return_name_flag:
        return model, this_model_name
    else:
        return model
#model_code:
#   0: default = keep avgpoool, no buffer
#   1: keep avgpoool, with buffer
#   2: remove avgpool, with buffer by default   
##### QC by --> print out model.keep_pool_flag and model.buffer_fc_flag    

#------------------------------------------------------------------------------------
# GAMR
#------------------------------------------------------------------------------------
def return_gamr_model_name_from_code(model_code):
    if model_code == 1:
        this_model_name = "GAMRwRN50"
    elif model_code == 2:
        this_model_name = "GAMRwRN50-PT"
    elif model_code == 3:
        this_model_name = "GAMR-tripleZ"
    elif model_code == 4:
        this_model_name = "GAMR-Psych"
    else:        
        this_model_name = "GAMRwConv"
    return this_model_name
    
    
def return_GAMR_model_by_code(model_code=4, gamr_time_step=4, return_name_flag=False):
    #For GAMR   ---> If model_code = 0 use the default encoder (Enc_Conv_v0_16)
    #For GAMR   ---> If model_code = 1 use the resnet50_block (pretrained=False) as encoder 
    #For GAMR   ---> If model_code = 2 use the resnet50_block (pretrained=True) as encoder 
    this_model = GAMR(time_step = gamr_time_step)
    if model_code == 1:
        this_model.encoder = Resnet_block_128OUT()
        this_model_name = "GAMRwRN50"
    elif model_code == 2:
        this_model.encoder = Resnet_block_128OUT(pretrained = True)
        this_model_name = "GAMRwRN50-PT"
    elif model_code == 3:
        this_model = GAMR_tripleZ(time_step = gamr_time_step)
        this_model_name = "GAMR-tripleZ"
    elif model_code == 4:
        this_model.encoder = Enc_Conv_Psych()
        this_model_name = "GAMR-Psych"
    else:
        this_model_name = "GAMR"
    if return_name_flag:
        return this_model, this_model_name
    else:
        return this_model

    
#=====================================================================================================================================================================================

########################################################
# Training Set-up
########################################################
import torch.optim as optim
# -----------------------------------------------------------
## General Training Set-up
# -----------------------------------------------------------
def retrieve_optimizer(model, optimizer_code, lr = 1e-3, momentum=0.9, weight_decay=0,  **kwargs):
    locals().update(kwargs) #bool(kwargs) = False is empty
    if optimizer_code == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_code == 'ADAMW': 
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)        
    else: 
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer
#optimizer = retrieve_optimizer(model, optimizer_code, **parameters_dict)

def retrieve_criterion(criterion_code):
    #-----------------------------
    # Criterion (loss function)
    #-----------------------------
    if criterion_code == 'CETP':
        criterion = nn.CrossEntropyLoss()
    elif criterion_code == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    elif criterion_code == 'MSE':
        criterion = nn.MSELoss()
    else:
        print("Warning: Unknown code for criterion --> use default = nn.CrossEntropyLoss()")
        criterion = nn.CrossEntropyLoss()
    return criterion
#criterion = retrieve_criterion(criterion_code = 'CETP')

#=====================================================================================================================================================================================
# -----------------------------------------------------------
## LR Scheduler
# -----------------------------------------------------------

#Note: For LR Schedule Hyperparameters
"""
default(the one that set on optimizer) learning_rate  = LR
Then, there are two variables: LR_M and LR_S
    Requirement: LR_M has to be < 1 (DEFAULT:0.1), LR_S has to be more than 2 (DEFAULT:4)

For 'CyclicLR'
    > set_CyclicLR_warmup(optimizer, base_lr = LR*LR_M, max_lr = LR, step_size_up = LR_S)
For 'CosineAnnealingLR'
    > set_CosineAnnealingLR(optimizer, T_max = LR_S*2, eta_min = LR*LR_M)
For StepLR
    > set_StepLR(optimizer, step_size=LR_S, gamma=LR_M*GM) where GM = 5

Reason: based on the visualization from https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863
"""
def set_StepLR(optimizer, step_size=7, gamma=0.1, **kwargs): 
    locals().update(kwargs)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return exp_lr_scheduler

def set_CyclicLR_warmup(optimizer, base_lr = 0.0001, max_lr = 1e-3, step_size_up = 4,  **kwargs):
    locals().update(kwargs)
    exp_lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, 
                     base_lr = base_lr, # Initial learning rate which is the lower boundary in the cycle for each parameter group
                     max_lr = max_lr, # Upper learning rate boundaries in the cycle for each parameter group
                     step_size_up = step_size_up, # Number of training iterations in the increasing half of a cycle
                     mode = "triangular2")
    return exp_lr_scheduler
#exp_lr_scheduler = set_CyclicLR_warmup(optimizer, base_lr = 0.0001, max_lr = 1e-3, step_size_up = 4)

def set_CosineAnnealingLR(optimizer, T_max = 8, eta_min = 1e-4, **kwargs ):
    locals().update(kwargs)
    #from optim.lr_scheduler import CosineAnnealingLR
    exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                  T_max = T_max, # Maximum number of iterations.
                                  eta_min = eta_min) # Minimum learning rate.
    return exp_lr_scheduler
#exp_lr_scheduler = set_CosineAnnealingLR(optimizer, T_max = 32, eta_min = 1e-4)

# -----------------------------------------------------------
## Validation Functions
# -----------------------------------------------------------
# This function have similar part with the train cell, see the explanation in the cell below
# In this function we use data test (testloaders) to validate the model result for each epoch, 
# and return an average loss and accuracy

def validate(this_model, criterion, loaders_for_data_to_val, num_val_sample = 1920, PRINT_FLAG=False):
    cnt_sample = 0
    this_model.eval()
    with torch.no_grad():
        total_loss = 0
        total_sample = 0    
        total_correct = 0        
        for image, label in loaders_for_data_to_val:
            image = image.to("cuda")
            label = label.to("cuda")
            out = this_model(image)
            loss = criterion(out, label)
            total_loss += loss.item()
            total_sample += len(label)
            total_correct += torch.sum(torch.max(out, 1)[1] == label).item()*1.0
            cnt_sample +=1
            if cnt_sample >= num_val_sample:
                break
    return  total_loss/total_sample, total_correct/total_sample

######################################################################################################################
##### MAIN SCRIPT
######################################################################################################################

def main():
    #####################################
    # PREPARATION (Declare variables and their presets)
    #####################################
    THIS_TASK = PRESET_THIS_TASK
    THIS_OBJ_CONFIG = PRESET_THIS_CONFIG
    THIS_ND =  PRESET_THIS_ND
    THIS_MODEL_CODE = DEFAULT_MODEL_CODE
    THIS_SCHEDULER =  PRESET_THIS_SCHEDULER #CONST, 'StepLR', 'AnnealingLR' #Note: 'CyclicLR' doesn't work with ADAM
    THIS_LR = PRESET_THIS_LR#1e-2, 1e-3, 1e-4, 1e-5
    THIS_LR_M = DEFAULT_LR_M
    THIS_LR_S = DEFAULT_LR_S
    THIS_SUFFIX = SUFFIX_RUN_CODE  
    THIS_VLD_SIZE = DEFAULT_NUM_VAL_SAMPLE
    THIS_NUM_EP = DEFAULT_MAX_EPOCH
    THIS_RANDOM_SEED = DEFAULT_RANDOM_SEED
    THIS_BATCH_SIZE = DEFAULT_BATCH_SIZE
    THIS_WEIGHT_DECAY = DEFAULT_WEIGHT_DECAY
    THIS_TIME_STEP = PRESET_GAMR_TIME_STEP
    THIS_OTMZ_CODE = DEFAULT_OPTIMIZER_CODE
    
    #====================================
    # For WANDB SWEEP
    #====================================
    if RUN_SWEEP:
        wandb.login()
        run = wandb.init(project=WANDB_PROJECT, reinit="True")
        THIS_NUM_EP = wandb.config.num_epochs
        THIS_SCHEDULER = wandb.config.scheduler_code
        THIS_WEIGHT_DECAY = wandb.config.weight_decay
        THIS_LR = wandb.config.LR
        THIS_LR_S = wandb.config.LR_S
        THIS_LR_M = wandb.config.LR_M



    #====================================
    # PARSER / Simulation Config
    #====================================

    # Initiate the parser
    parser = argparse.ArgumentParser()

    # Add long and short argument
    parser.add_argument("--task", "-t", help="specify task ('AB' or 'SD')")
    parser.add_argument("--obj_config", "-o", help="specify type of object's role information ('SA', 'LOC', or 'ID')")
    parser.add_argument("--num_distractor", "-d",  help="specify the number of distractors (n items)")
    parser.add_argument("--model_code", "-i", help="Model Code (0:default, 1:add buffer, 2:remove avgpool)")
    #model_code:
    #   0: default = keep avgpoool, no buffer
    #   1: keep avgpoool, with buffer
    #   2: remove avgpool, with buffer by default   
    parser.add_argument("--scheduler_code", "-H", help="Code for LR-Scheduler ('CONST', 'StepLR', 'AnnealingLR')")
    parser.add_argument("--LR", "-L", help="Learning Rate and/or LR for scheduler")
    parser.add_argument("--LR_M", "-M", help="M(lr multiplicative variable) for LR scheduler")
    parser.add_argument("--LR_S", "-S", help="S(step size variable) for LR scheduler")
    
    parser.add_argument("--weight_decay", "-w", help="weight_decay (L2) for ADAM")
    parser.add_argument("--time_step", "-T", help="Time Step for GAMR")

    parser.add_argument("--suffix", "-f", help="add suffix for the run")
    parser.add_argument("--vld_size", "-v", help="Number of sample size for the validation set")
    parser.add_argument("--num_epochs", "-p", help="Number of epoches")
    parser.add_argument("--batch_size", "-b", help="Batch Size")
    
    parser.add_argument("--optimizer_code", "-Z", help="Specify Optimizer")  
    parser.add_argument("--random_seed", "-R", help="Specify seed for all random functions")
    
    args = parser.parse_args()

    wandb_config = {'TIMESTAMP':get_time_stamp(mode='date_time')}

    # Check 
    if args.task:
        THIS_TASK=args.task
    wandb_config['ThisTask'] = THIS_TASK #Main info -- always include
    if args.obj_config:
        THIS_OBJ_CONFIG=args.obj_config  
    wandb_config['ObjConfig'] = THIS_OBJ_CONFIG #Main info -- always include
    if args.num_distractor:
        THIS_ND = int(args.num_distractor)
    wandb_config['NumDTT'] = THIS_ND #Main info -- always include
    if args.model_code:
        THIS_MODEL_CODE = int(args.model_code)
        wandb_config['ModelCode'] = THIS_MODEL_CODE
    if args.scheduler_code:
        THIS_SCHEDULER = args.scheduler_code
        wandb_config['ScdlCode'] = THIS_SCHEDULER
    if args.weight_decay:
        THIS_WEIGHT_DECAY = float(args.weight_decay)
        wandb_config['WD'] = THIS_WEIGHT_DECAY
    if args.time_step:
        THIS_TIME_STEP = int(args.time_step)
        wandb_config['TimeSTEP'] = THIS_TIME_STEP     
    if args.LR:
        THIS_LR = float(args.LR)
        wandb_config['LR'] = THIS_LR
    if args.LR_M:
        THIS_LR_M = float(args.LR_M)
        wandb_config['LR_M'] = THIS_LR_M
    if args.LR_S:
        THIS_LR_S = int(args.LR_S)
        wandb_config['LR_S'] = THIS_LR_S
    if args.suffix:
        THIS_SUFFIX = args.suffix
    if args.vld_size:
        THIS_VLD_SIZE = int(args.vld_size)
        wandb_config['vldSz'] = THIS_VLD_SIZE
    if args.num_epochs:
        THIS_NUM_EP = int(args.num_epochs)
        wandb_config['numEP'] = THIS_NUM_EP
    if args.random_seed:
        THIS_RANDOM_SEED = int(args.random_seed)
    wandb_config['SEED'] = THIS_RANDOM_SEED
    if args.batch_size:
        THIS_BATCH_SIZE = int(args.batch_size)
        wandb_config['BatchSz'] = THIS_BATCH_SIZE
    if args.optimizer_code:
        THIS_OTMZ_CODE = args.optimizer_code
        wandb_config['OptmzCode'] = THIS_OTMZ_CODE

    # total arguments
    n = len(sys.argv)
    print("Total arguments passed:", n)   

    # Arguments passed
    print("\nName of Python script:", sys.argv[0])   
    print("\nArguments passed:", end = " ")
    for i in range(1, n):
        print(sys.argv[i], end = " ")
    print("\n")

    if RECORD_HOST_NAME_FLAG:
        HOST = get_host_name()
        print(f"HOST MACHINE: {HOST}\n")
        wandb_config['HOST'] = HOST

    #====================================
    # Get Running Code and Record Configuration File
    #====================================
    running_code = generate_running_code_from_config_dict(wandb_config,  suffix_run_code = THIS_SUFFIX)
    wandb_config['ThisRunCode'] = running_code

    #######################################################
    # SET-UP Parameters for this run
    ########################################################
    dataset_name = DATASET_NAME
    #---------------------------------------------------------------
    ### Experiment Set-Up
    #---------------------------------------------------------------
    task_name = THIS_TASK
    obj_config_type = THIS_OBJ_CONFIG
    num_distractor = THIS_ND
    model_code = THIS_MODEL_CODE

    frozen_features_flag = DEFAULT_FROZEN_FLAG
    wandb_config['frozen_features_flag'] = frozen_features_flag
    
    all_random_seed = THIS_RANDOM_SEED

    #---------------------------------------------------------------
    ### General Training Set-Up
    #---------------------------------------------------------------
    train_with_more_data_flag = DEFAULT_MORE_DATA_FLAG
    grad_clip_val = DEFAULT_GRAD_CLIP_VAL
    batch_size = THIS_BATCH_SIZE
    criterion_code = DEFAULT_CRITERION_CODE
    optimizer_code = THIS_OTMZ_CODE
    num_validation_sample = THIS_VLD_SIZE
    num_epochs = THIS_NUM_EP
    shuffle_flag = DEFAULT_SHUFFLE_FLAG
    weight_decay = THIS_WEIGHT_DECAY
    gamr_time_step = int(THIS_TIME_STEP)
    #---------------------------------------------------------------
    ### LR Scheduler Set-Up
    #---------------------------------------------------------------
    LR = THIS_LR
    LR_scheduler_code = THIS_SCHEDULER
    LR_M = THIS_LR_M
    LR_S = THIS_LR_S
    
    #---------------------------------------------------------------
    ### Get Model's name
    #---------------------------------------------------------------
    if MODEL_NAME.startswith('Res'):
        this_model_name = return_resnet_model_name_from_code(model_code, frozen_features_flag, baseRN_type=50)
    elif MODEL_NAME.startswith('GAMR'):
        this_model_name = return_gamr_model_name_from_code(model_code)
    else:
        this_model_name = MODEL_NAME
    model_name = this_model_name
    wandb_config['model_name'] = model_name 
    
    ########################################################
    # PATHS
    ########################################################
    path_to_save_model_state = PATH_TO_SAVE_MODEL_STATE
    path_to_save_log_file = PATH_TO_SAVE_LOG_FILE
    ########################################################
    # SET SEED
    ########################################################
    set_all_random_seed(input_seed=all_random_seed)
    set_all_torch_random_seed(seed=all_random_seed)   
    
    ########################################################
    # Logging information of this run
    ########################################################

    #====================================
    # Get Running Code and Record Configuration File
    #====================================


    if not LOCAL_TEST_MODE and not RUN_SWEEP:
        wandb.login()
        run = wandb.init(project=WANDB_PROJECT, reinit="True", config=wandb_config)

    #---------------------------------------------------------------
    ### INTERNAL LOG
    #---------------------------------------------------------------
    internal_log = {'TIMESTAMP':get_time_stamp(mode='date_time')}
    internal_log['running_code'] = running_code
    internal_log['model_name'] = model_name
    internal_log['dataset_name'] = dataset_name    
    internal_log['train_with_more_data_flag'] = train_with_more_data_flag    
    internal_log['task_name'] = task_name
    internal_log['obj_config_type'] = obj_config_type
    internal_log['num_distractor'] = num_distractor
    internal_log['model_code'] = model_code
    internal_log['grad_clip_val'] = grad_clip_val    
    internal_log['frozen_features_flag'] = frozen_features_flag
    internal_log['all_random_seed'] = all_random_seed
    internal_log['batch_size'] = batch_size
    internal_log['criterion_code'] = criterion_code
    internal_log['optimizer_code'] = optimizer_code
    internal_log['weight_decay'] = weight_decay
    internal_log['num_validation_sample'] = num_validation_sample
    internal_log['num_epochs'] = num_epochs
    internal_log['shuffle_flag'] = shuffle_flag
    internal_log['LR'] = LR
    internal_log['LR_scheduler_code'] = LR_scheduler_code
    internal_log['LR_M'] = LR_M
    internal_log['LR_S'] = LR_S
    internal_log['GAMR_T'] = gamr_time_step
    internal_log['path_to_save_model_state'] = path_to_save_model_state
    internal_log['path_to_save_log_file'] = path_to_save_log_file

    ########################################################
    # Report to user
    ########################################################
    print("=============================================================================================")
    print(f"\tTASK:{task_name}, ObjConfig:{obj_config_type}, NumDistractors:{num_distractor}\t")
    print("=============================================================================================")
    print("---------------------------------------------------------------------------------------------")
    ########################################################
    # The DATASET
    ########################################################

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(IMGNET_MEAN, IMGNET_STD)
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(IMGNET_MEAN,  IMGNET_STD)
    ])

    train_set_data = HEPT4MD_CustomizeDataset_byALPHABET(
        dict_of_exp_info = {
            'task_name': task_name,  #'SD', 'LR', 'AB'
            'obj_config':obj_config_type, # 'SA' , 'ID', 'LOC'
            'num_distractor':num_distractor,       # 0, 1, 2, 3
            'set_type':'A'      #
        },
        transform=train_transform, 
        internal_shuffle_flag = INTERNAL_SHUFFLE_FLAG
    )

    dev_set_data = HEPT4MD_CustomizeDataset_byALPHABET(
        dict_of_exp_info = {
            'task_name': task_name,  #'SD', 'LR', 'AB'
            'obj_config':obj_config_type, # 'SA' , 'ID', 'LOC'
            'num_distractor':num_distractor,       # 0, 1, 2, 3
            'set_type':'C'      #
        },
        transform=test_transform, 
        internal_shuffle_flag = INTERNAL_SHUFFLE_FLAG
    )


    train_dataloader = DataLoader(train_set_data, batch_size=batch_size, shuffle=shuffle_flag)
    validation_dataloader = DataLoader(dev_set_data, batch_size=batch_size, shuffle=shuffle_flag)  
    print(f"\t Path to Dataset - Train Set:\n\t\t{train_set_data.path_to_folder}\n")
    print(f"\t Path to Dataset - Development Set:\n\t\t{dev_set_data.path_to_folder}\n")
    print(f"\t batch_size:{batch_size}, shuffle_flag:{shuffle_flag}")
    internal_log['path_to_data_train_set'] = train_set_data.path_to_folder
    internal_log['path_to_data_development_set'] = dev_set_data.path_to_folder

    #If double training data
    if train_with_more_data_flag:
        train_set_data_more = HEPT4MD_CustomizeDataset_byALPHABET(
            dict_of_exp_info = {
                'task_name': task_name,  #'SD', 'LR', 'AB'
                'obj_config':obj_config_type, # 'SA' , 'ID', 'LOC'
                'num_distractor':num_distractor,       # 0, 1, 2, 3
                'set_type':'D'      #
            },
            transform=train_transform, 
            internal_shuffle_flag = INTERNAL_SHUFFLE_FLAG
        )
        more_train_dataloader = DataLoader(train_set_data_more, batch_size=batch_size, shuffle=shuffle_flag)
        print(f"\t Path to Dataset - \'MORE\' - Train Set:\n\t\t{train_set_data_more.path_to_folder}\n")
        internal_log['path_to_data_train_more_set'] = train_set_data_more.path_to_folder

    print("---------------------------------------------------------------------------------------------")

    if LOCAL_TEST_MODE:
        import pdb; pdb.set_trace()

    #########################################################################
    # RUN SET-UP
    ########################################################################
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set up GPU device for pytorch

    signal.signal(signal.SIGINT, handler) #in case someone press ctrl+c  by mistake 

    ########################################################
    # Training Set-up
    ########################################################  

    if MODEL_NAME.startswith('GAMR'):
        this_model = return_GAMR_model_by_code(model_code = model_code, gamr_time_step = gamr_time_step)
    else:
        this_model = return_RN_model_by_code(model_code = model_code, frozen_features_flag = frozen_features_flag)

    this_model = this_model.to("cuda")
    criterion = retrieve_criterion(criterion_code = criterion_code)
    optimizer = retrieve_optimizer(this_model, optimizer_code=optimizer_code, lr = LR, weight_decay=weight_decay)

    print(f"\t [{this_model_name}] model_code:{model_code}, criterion_code:{criterion_code}, optimizer_code:{optimizer_code}, LR:{LR:.1E}")
    print("---------------------------------------------------------------------------------------------")
    #---------------------------------------------------------------
    ### LR Scheduler Selection
    #---------------------------------------------------------------   
    if LR_scheduler_code == 'CONST':
        print(f"Use Constant Learning Rate, CODE = {LR_scheduler_code}")
        exp_lr_scheduler = None
    else: #Use LR_Scheduler
        print(f"Use Learning Rate Scheduler, CODE = {LR_scheduler_code}")
        print(f"\t LR_M:{LR_M}, LR_S:{LR_S}")
        if LR_scheduler_code == 'CyclicLR': 
            exp_lr_scheduler = set_CyclicLR_warmup(optimizer, base_lr = LR*LR_M, max_lr = LR, step_size_up = LR_S)
            print(f"\t\t base_lr = {LR*LR_M:.1E}, max_lr = {LR:.1E}, step_size_up = {LR:.1E}")
        elif LR_scheduler_code == 'AnnealingLR':
            exp_lr_scheduler = set_CosineAnnealingLR(optimizer, T_max = LR_S*2, eta_min = LR*LR_M)
            print(f"\t\t  T_max = {LR_S*2}, eta_min = {LR*LR_M:.1E}")
        else: #Default = 'StepLR'
            exp_lr_scheduler = set_StepLR(optimizer, step_size=LR_S, gamma=LR_M*5)
            print(f"\t\t step_size={LR_S}, gamma={LR_M*5}")
    print("---------------------------------------------------------------------------------------------")

    if LOCAL_TEST_MODE:
        import pdb; pdb.set_trace()
    ########################################################
    # Main Loops
    ########################################################   
    #---------------------------------------------------------------
    ### Debugging set-up and final report to user
    #---------------------------------------------------------------  
    if LOCAL_TEST_MODE or RUN_WandB_as_scratch: #For Debugging
        num_epochs = LR_S*2 + 1
        total_num_batch = 10
        num_validation_sample = batch_size
    else:
        if train_with_more_data_flag:
            total_num_batch = 2*len(train_dataloader)
        else:
            total_num_batch = len(train_dataloader)

    print(f"\t[ACTUAL] num_epochs:{num_epochs}, total_num_batch:{total_num_batch}, num_validation_sample:{num_validation_sample}")
    print("---------------------------------------------------------------------------------------------")
    #---------------------------------------------------------------
    ### WandB Set-up
    #---------------------------------------------------------------   
    if not LOCAL_TEST_MODE:
        if WANDB_LOG_ALL_FLAG:
            #For Real exp
            wandb.watch(this_model, criterion, log="all", log_freq=20) # WANDB WATCH
        else:
            #For HP search
            wandb.watch(this_model, criterion, log="gradients", log_freq=total_num_batch) # WANDB WATCH    
    print("=============================================================================================")
    if not AUTORUN:
        import pdb; pdb.set_trace()
    #---------------------------------------------------------------
    ### Logging start time
    #---------------------------------------------------------------         
    import time
    from datetime import timedelta
    from datetime import datetime
    start_time = time.monotonic()
    print(f"Start running at {datetime.now()}")
    print("=============================================================================================")    
    #---------------------------------------------------------------
    ### START
    #---------------------------------------------------------------  
    cnt_b = 0
    this_lr = LR
    hist_val_acc = []
    hist_train_acc = []
    hit_ceiling_flag = False
    curr_at_ceil_flag = False
    cnt_ceiling_hit = 0
    for ep_i in tqdm(range(num_epochs)):
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
        this_model.train()
        total_loss = 0
        total_sample = 0
        total_correct = 0
        b_i =0 
        if train_with_more_data_flag:
            for image, label in more_train_dataloader: 
                image = image.to(DEVICE) 
                label = label.to(DEVICE) 
                
                out = this_model(image) # STEP 1: forward propagation
                loss = criterion(out, label) # STEP 2: calculate loss
                optimizer.zero_grad() # STEP 3: zero out the gradient. see: https://stackoverflow.com/q/48001598/2147347
                loss.backward() # STEP 4: backpropagation
                torch.nn.utils.clip_grad_norm_(this_model.parameters(), grad_clip_val) #STEP 5: clip gradients to prevent nan
                optimizer.step() # STEP 6: update the model
                #gr_model.g_out.weight
                total_loss += loss # sum of losses for this epoch
                total_sample += len(label) # number seen images in this epoch
                total_correct += torch.sum(torch.max(out,1)[1]==label)*1.0 # sum of the correct prediction
                
                curr_acc = total_correct/total_sample
                if not LOCAL_TEST_MODE:
                    wandb.log({"curr_train_acc": total_correct/total_sample, "curr_train_loss":total_loss/total_sample}, step=cnt_b)
                else: #print locally
                    print(f"Training Batch#{cnt_b} of Epoch#{ep_i} (num batch per ep = {total_num_batch}) | LR={this_lr} Loss={total_loss/total_sample}, Training ACC = {total_correct/total_sample}")
                    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
                metrics = {"train/train_loss": loss.item(), 
                       "train/train_acc": curr_acc,
                       "train/batch_cnt":cnt_b}
                cnt_b +=1
                b_i+=1
                if b_i >= total_num_batch: #for testing only
                    break        
        for image, label in train_dataloader: 
            image = image.to(DEVICE) 
            label = label.to(DEVICE) 
            
            out = this_model(image) # STEP 1: forward propagation
            loss = criterion(out, label) # STEP 2: calculate loss
            optimizer.zero_grad() # STEP 3: zero out the gradient. see: https://stackoverflow.com/q/48001598/2147347
            loss.backward() # STEP 4: backpropagation
            optimizer.step() # STEP 5: update the model
            #gr_model.g_out.weight
            total_loss += loss # sum of losses for this epoch
            total_sample += len(label) # number seen images in this epoch
            total_correct += torch.sum(torch.max(out,1)[1]==label)*1.0 # sum of the correct prediction
            
            curr_acc = total_correct/total_sample
            if not LOCAL_TEST_MODE:
                wandb.log({"curr_train_acc": total_correct/total_sample, "curr_train_loss":total_loss/total_sample}, step=cnt_b)
            else: #print locally
                print(f"Training Batch#{cnt_b} of Epoch#{ep_i} (num batch per ep = {total_num_batch}) | LR={this_lr} Loss={total_loss/total_sample}, Training ACC = {total_correct/total_sample}")
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
            metrics = {"train/train_loss": loss.item(), 
                   "train/train_acc": curr_acc,
                   "train/batch_cnt":cnt_b}
            cnt_b +=1
            b_i+=1
            if b_i >= total_num_batch: #for testing only
                break
   
        loss = total_loss/total_sample # averaging loss
        acc = total_correct/total_sample # averaging accuracy
        hist_train_acc.append(acc.item())

        val_loss, val_acc = validate(this_model, criterion, loaders_for_data_to_val=validation_dataloader, num_val_sample = num_validation_sample) # validate using data "test" at the end of an epoch
        hist_val_acc.append(val_acc)

        if exp_lr_scheduler:
            this_lr = exp_lr_scheduler.get_last_lr()[0]
            exp_lr_scheduler.step()
        else:
            this_lr = optimizer.defaults['lr']


        val_metrics = {"train/final_loss":loss,
        "train/final_acc":acc,
        "train/this_lr":this_lr,
        "val/val_accuracy": val_acc,
        "val/val_loss": val_loss,
        "val/epoch_cnt":ep_i
        }
        print(f"\n[Task:{task_name} ObjConfig:{obj_config_type} ND:{num_distractor}] | epoch {ep_i}, Last LR={this_lr} | loss:{loss}, acc:{acc}, val_loss: {val_loss}, val_acc:{val_acc}\n") # our personal log        
        if not LOCAL_TEST_MODE:
            #wandb.log({"final_loss":loss, "final_acc":acc, "val_loss": val_loss, "val_acc":val_acc}, step=i) # WANDB LOG
            wandb.log({**metrics, **val_metrics})
        else:
            import pdb; pdb.set_trace()
        
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
        #------------------------------------------------------------------------------
        # Checking early stop condition on hitting ceiling fopr training accuracy

        if acc >= CEILING_OF_TRAIN_ACC and not (curr_at_ceil_flag):
            hit_ceiling_flag = True
            curr_at_ceil_flag = True
            cnt_ceiling_hit = 1
        elif acc >= CEILING_OF_TRAIN_ACC and val_acc >= CONDITION_VAL_ACC_OF_CEILING_TRAIN: #and curr_at_ceil_flag = True
            cnt_ceiling_hit += 1
        else: #acc < 1.0
            curr_at_ceil_flag = False
            cnt_ceiling_hit = 0

        if curr_at_ceil_flag and (cnt_ceiling_hit >= CNT_NUM_CEILING_BEFORE_STOP):
            #Early Stop
            print(f"Training Accuracy hit ceiling of {CEILING_OF_TRAIN_ACC} for {cnt_ceiling_hit} epoch > {CNT_NUM_CEILING_BEFORE_STOP}")
            print(f"Early stop training at num_epoch = {ep_i+1}")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
            break
        #--------------------------------    
        # Checking early stop condition on hitting ceiling for validation accuracy
        if val_acc >= EARLY_STOP_VAL_ACC:
            print(f"Reach validation accuracy of {val_acc} <= {EARLY_STOP_VAL_ACC}")
            print(f"Early stop training at num_epoch = {ep_i+1}")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
            break
    #---------------------------------------------------------------
    ### RUN END
    #---------------------------------------------------------------          
    print("=============================================================================================")  
    print(f"Finish running at {datetime.now()}")
    end_time = time.monotonic()
    print(f"\t Runtime Duration: {timedelta(seconds=end_time - start_time)}")
    internal_log['run_time_duration'] = f"{timedelta(seconds=end_time - start_time)}"

    print("=============================================================================================")  
    #---------------------------------------------------------------
    ### Record rough results
    #---------------------------------------------------------------  
    internal_log['last_training_acc'] = acc.item()
    internal_log['last_training_loss'] = loss.item()
    internal_log['final_val_acc'] = val_acc
    internal_log['final_val_loss'] = val_loss
    internal_log['hist_val_acc'] = hist_val_acc
    internal_log['hist_train_acc'] = hist_train_acc
    

    if LOCAL_TEST_MODE:
        print(f"History of Validation Accuracy: {[f'{va:.4f}' for va in hist_val_acc]}\n")
        print(f"History of Last Training Accuracy: {[f'{ta:.4f}' for ta in hist_train_acc]}\n")
        import pdb; pdb.set_trace()


    #---------------------------------------------------------------
    ### Test full validation set (or test set in final evaluation)
    #---------------------------------------------------------------  
    if LOCAL_TEST_MODE or RUN_WandB_as_scratch: #For Debugging
        total_test_sample = batch_size
    else:
        total_test_sample = len(validation_dataloader)

    test_loss, test_acc = validate(this_model, criterion, loaders_for_data_to_val=validation_dataloader, num_val_sample = total_test_sample)
    internal_log['test_loss'] = test_loss
    internal_log['test_acc'] = test_acc

    print(f"Testing on the full validation set (#sample = {total_test_sample}):")
    print(f"\t\t Loss:{test_loss:.8}, Accuracy:{test_acc:.6}")
    print("=============================================================================================")  

    if not LOCAL_TEST_MODE:
        wandb.log({"test_loss": test_loss, "test_acc":test_acc})
    
    #---------------------------------------------------------------
    ### Save Model state_dict()
    #---------------------------------------------------------------  
    if not LOCAL_TEST_MODE:
        # =========================================
        # Save the model, save it to wandb server
        # =========================================
        name_to_save_model = f"{running_code}__AtEpoch_{ep_i}.pt"
        path_to_model = f"{path_to_save_model_state}{name_to_save_model}"
        torch.save(this_model.state_dict(),path_to_model)
        wandb.save(path_to_model)    
        # =========================================
        # Finish the tracking
        # =========================================
        #If you want to make another run, marks the current run as finished. W&B will finish uploading the data and log others information such a packages installed, wandb config files, etc.

        run.finish()
        print(f"Close wandb at {datetime.now()}")
        print("=============================================================================================")  

    #===============================================================================
    # Save Log Files
    #===============================================================================
    if not LOCAL_TEST_MODE: #Only save the real run
        name_to_save_log_file = f"{running_code}__AtEpoch_{ep_i}.txt"
        with open(path_to_save_log_file+name_to_save_log_file, 'w') as log_file:
             log_file.write(json.dumps(internal_log))
        print(f"Save internal log file as {name_to_save_log_file} \n\t\t at {path_to_save_log_file}\n")
    if LOCAL_TEST_MODE:
        printout_dict(internal_log, desc='INTERNAL LOG')
        import pdb; pdb.set_trace()
    #===============================================================================
    ########################################################
    # MAIN LOOP DONE!
    ########################################################  
    print("=============================================================================================")   

if RUN_SWEEP:
    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=num_run_sweep)

if __name__ == '__main__':
    main()
