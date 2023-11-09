"""
	List of defaults parameters and some presets

	#exec(open("defaults_parameters.py").read()) 
"""


######################################################################################################################
### PREPARATION: DEFAULT PARAMETERS
######################################################################################################################
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD =[0.229, 0.224, 0.225]
DEFAULT_RANDOM_SEED = 778899
#-------------------------------------------
# Defaults Model and Training Set-Up
#-------------------------------------------
DEFAULT_FROZEN_FLAG = True
DEFAULT_MAX_EPOCH = 100
DEFAULT_BATCH_SIZE = 60
DEFAULT_CRITERION_CODE = 'CETP'
DEFAULT_OPTIMIZER_CODE = 'ADAM'
DEFAULT_SCHEDULER_CODE = 'CONST'
DEFAULT_LR = 2.5e-4
DEFAULT_LR_M = 0.1
DEFAULT_LR_S = 4

DEFAULT_NUM_VAL_SAMPLE = 1920
DEFAULT_EARLY_STOP_VAL_ACC = 0.9999
DEFAULT_SHUFFLE_FLAG = False

#-------------------------------------------
# For configuration
#-------------------------------------------
RECORD_HOST_NAME_FLAG = True
