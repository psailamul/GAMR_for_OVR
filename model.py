"""
Where all models classes are
"""
#exec(open("model.py").read())
"""
Models List
- resnet_unify(self, baseRN = 50, frozen_features_flag = True, keep_pool_flag=True, buffer_fc_flag = False, add_sigmoid_flag=False)
- triplresnet_unify(self, numHead=3, keep_pool_flag=True, buffer_fc_flag = True, baseRN = 18 )
"""

###############################################################################################################
# IMPORT 
###############################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy


###############################################################################################################
# CLASSES
###############################################################################################################

#-------------------------------------------------------------------------------------------------------
# Class for Model = resnet_unify
#-------------------------------------------------------------------------------------------------------
class resnet_unify(nn.Module):
    def __init__(self, baseRN = 50, frozen_features_flag = True, keep_pool_flag=True, buffer_fc_flag = False, add_sigmoid_flag=False, *args, **kwargs):
        super(resnet_unify, self).__init__()
        self.frozen_features_flag = frozen_features_flag
        self.keep_pool_flag = keep_pool_flag
        self.buffer_fc_flag = buffer_fc_flag
        self.add_sigmoid_flag = add_sigmoid_flag
        self.RNbase = baseRN #18, 34, 50
        # Get pretrained Resnet
        if self.RNbase >= 50:
            resnet_pretrained_model = models.resnet50(pretrained=True)  
            last_output = 2048
            last_filter = 7    #Note: 2048*7*7 = 100352 
            buffer_FC_num_unit = last_output / 2 #1024
        elif self.RNbase == 34:
            resnet_pretrained_model = models.resnet34(pretrained=True)        
            last_output = 512
            last_filter = 7    #Note: 512*7*7 = 25088
            buffer_FC_num_unit = last_output / 2 #256
        else:
            resnet_pretrained_model = models.resnet18(pretrained=True)         #default
            last_output = 512
            last_filter = 7
            buffer_FC_num_unit = last_output / 2 #256
        self.NumUnitBufferFC = int(buffer_FC_num_unit/2)
        #Freeze weights or not
        if self.frozen_features_flag:
            for param in resnet_pretrained_model.parameters():
                param.requires_grad = False 
        #Include the Average pool layer or not
        RN_in_list = list(resnet_pretrained_model.children())  
        #before_avg_pool 
        layers_before_avg_pool  = RN_in_list[:-2]
        my_features_layers = copy.deepcopy(layers_before_avg_pool)
        if keep_pool_flag: #Keep average pool layer or not --> replace .fc and .avepool
            my_features_layers.append(RN_in_list[8])
        self.features_extractor = torch.nn.Sequential(*my_features_layers)
        #--------------------------------------
        #Manipulate FC for transfer Learning
        #--------------------------------------
        #When there is no avg pool -- really large output  -- add one large buffer layer by default
        #When there is average pool --> buffer layer size = half of the last_output
        if bool(kwargs):
            if 'NumUnitBufferFC' in kwargs.keys():
                self.NumUnitBufferFC= int(kwargs['NumUnitBufferFC'])

        self.last_output = last_output
        self.last_filter = last_filter

        if keep_pool_flag: #If there is avgpool --> last output as stated (basicly the same size)

            if buffer_fc_flag:
                self.fc = nn.Linear(self.last_output, int(self.last_output/2))      #Ex for 50 --> 2048 x 1048
                self.fc2 = nn.Linear(int(self.last_output/2), self.NumUnitBufferFC) #Ex for 50 --> 1048 x 512
                nn.init.xavier_uniform_(self.fc.weight)   #Ex for 34 --> 512 x 256
                nn.init.xavier_uniform_(self.fc2.weight)  #Ex for 34 --> 256 x 128
                self.fc.bias.data.fill_(0.01)
                self.fc2.bias.data.fill_(0.01)
            else: 
                self.fc = nn.Linear(self.last_output, self.NumUnitBufferFC) #Ex for 50 --> 2048 x 512
                nn.init.xavier_uniform_(self.fc.weight)   #Ex for 34 --> 512 x 128
                self.fc.bias.data.fill_(0.01)
        else: #When there is no avg pool -- really large output  
            # -- add one large buffer layer by default with the same size as last_output
            self.fc = nn.Linear(self.last_output*last_filter*last_filter, self.last_output)  #Ex for 50 --> 100352 x 2048
            self.fc.bias.data.fill_(0.01) #Ex for 34 --> 25088 x 512

            self.buffer_fc_flag = True #add by default

            self.fc2 = nn.Linear(self.last_output, self.NumUnitBufferFC) #Ex for 50 --> 2048 x 512
            nn.init.xavier_uniform_(self.fc2.weight) #Ex for 34 --> 512 x 128
            self.fc2.bias.data.fill_(0.01)
                
        self.bn_class = nn.Linear(self.NumUnitBufferFC, 2) #Ex for 50 --> 512 x 2
        nn.init.xavier_uniform_(self.bn_class.weight) #Ex for 34 --> 128 x 2
        self.bn_class.bias.data.fill_(0.01)

    def forward(self, X):        
        x = self.features_extractor(X)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc(x)) 
        if self.buffer_fc_flag:
            x = F.relu(self.fc2(x))           
        x = self.bn_class(x)
        if self.add_sigmoid_flag: #Default = False               
            x = torch.sigmoid(x)
        return x

#-------------------------------------------------------------------------------------------------------
# Class for Model = triplresnet
#-------------------------------------------------------------------------------------------------------
class triplresnet_unify(nn.Module):
    def __init__(self, numHead=3, keep_pool_flag=True, buffer_fc_flag = True, baseRN = 18 ):
        super(triplresnet_unify, self).__init__()
        self.NumHead = numHead
        self.keep_pool_flag = keep_pool_flag
        self.buffer_fc_flag = buffer_fc_flag
        self.RNbase = baseRN #18, 34, 50
        if self.RNbase >= 50:
            resnet_pretrained_model = models.resnet50(pretrained=True)  
            last_output = 2048
            last_filter = 7     
        elif self.RNbase == 34:
            resnet_pretrained_model = models.resnet34(pretrained=True)        
            last_output = 512
            last_filter = 7
        else:
            resnet_pretrained_model = models.resnet18(pretrained=True)         #default
            last_output = 512
            last_filter = 7
    
        RN_in_list = list(resnet_pretrained_model.children())        
        conv1_weight = resnet_pretrained_model.conv1.weight
        mean_conv1_weight = torch.mean(conv1_weight, dim=1, keepdim=True)
        np_w = mean_conv1_weight.detach().numpy()
        one_ch_conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            one_ch_conv1.weight = nn.Parameter(torch.from_numpy(np_w).float())
        one_ch_RN_trunk = []
        one_ch_RN_trunk.append(one_ch_conv1)
        for l in  range(1, 8):
            one_ch_RN_trunk.append(RN_in_list[l])
        #from conv1 to before pool
        if keep_pool_flag:
            one_ch_RN_trunk.append(RN_in_list[8])
        self.one_trunk = torch.nn.Sequential(*one_ch_RN_trunk)
        self.heads = [] 
        for i in range(self.NumHead):
            self.heads.append(self.one_trunk)
        #For RN50: last output = xxx ? 
        #RuntimeError: mat1 and mat2 shapes cannot be multiplied (25x6144 and 1536x512)
        self.NumUnitFCextralarge = 2048
        self.NumUnitFClarge = 1024
        self.NumUnitFCsmall = 512

        self.last_output = last_output
        self.last_filter = last_filter
        if keep_pool_flag:
            if buffer_fc_flag:
                self.fc1 = nn.Linear(self.NumHead*self.last_output, self.NumUnitFClarge) #[ 1536, 1024]          
                self.fc2 = nn.Linear(self.NumUnitFClarge, self.NumUnitFCsmall)
                nn.init.xavier_uniform_(self.fc1.weight)
                nn.init.xavier_uniform_(self.fc2.weight)
                self.fc1.bias.data.fill_(0.01)
                self.fc2.bias.data.fill_(0.01)
            else: 
                self.fc1 = nn.Linear(self.NumHead*self.last_output, self.NumUnitFCsmall)
                nn.init.xavier_uniform_(self.fc1.weight)
                self.fc1.bias.data.fill_(0.01)
        else:  #keep_pool_flag = False --> last output = 512 ch of 7x7
            if buffer_fc_flag:
                self.fc1 = nn.Linear(self.NumHead*self.last_output*last_filter*last_filter, self.NumUnitFClarge) #[75264, 1024]
                self.fc2 = nn.Linear(self.NumUnitFClarge, self.NumUnitFCsmall)
                nn.init.xavier_uniform_(self.fc1.weight)
                nn.init.xavier_uniform_(self.fc2.weight)
                self.fc1.bias.data.fill_(0.01)
                self.fc2.bias.data.fill_(0.01)
            else: 
                self.fc1 = nn.Linear(self.NumHead*self.last_output*last_filter*last_filter, self.NumUnitFCsmall)
                nn.init.xavier_uniform_(self.fc1.weight)
                self.fc1.bias.data.fill_(0.01)
        self.bn_class = nn.Linear(self.NumUnitFCsmall, 2) 
        nn.init.xavier_uniform_(self.bn_class.weight)
        self.bn_class.bias.data.fill_(0.01)
    def forward(self, X): 
             
        assert X.shape[1] == self.NumHead
        Xs = []
        for i in range(self.NumHead):
            this_X = X[:,i,:,:]
            this_X = torch.unsqueeze(this_X,1)
            Xs.append(self.heads[i](this_X))
        #import pdb; pdb.set_trace()
        x = torch.cat(Xs, dim=1) 
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))                     
        if self.buffer_fc_flag:
            x = F.relu(self.fc2(x))           
        x = self.bn_class(x)                  
        x = torch.sigmoid(x)
        return x

#------------------------------------
# Model
#------------------------------------
#model1 = triplresnet_unify(keep_pool_flag=True, buffer_fc_flag=True, baseRN = 50)
#model2 = triplresnet_unify(keep_pool_flag=True, buffer_fc_flag=False, baseRN = 50)
#model3 = triplresnet_unify(keep_pool_flag=False, buffer_fc_flag=True, baseRN = 50)
#model4 = triplresnet_unify(keep_pool_flag=False, buffer_fc_flag=False, baseRN = 50)
#dummy_input_3ch = torch.rand((50, 3, 224, 224))
#dummy_input_1ch = torch.rand((50, 1, 224, 224))
#outputs1 = model1(dummy_input_3ch)
#outputs2 = model2(dummy_input_3ch)
#outputs3 = model3(dummy_input_3ch)
#outputs4 = model4(dummy_input_3ch)
# OneCH_RN_triplets_FC_converge  == triplresnet