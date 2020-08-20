#!/usr/bin/env python

import numpy as np, os, sys, joblib
from scipy.io import loadmat
from get_12ECG_features import get_12ECG_features
import pandas as pd
import os,time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader,Dataset
from config import config
import utils
import warnings
import joblib
from sklearn import metrics

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(666)
torch.cuda.manual_seed(666)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,padding=3, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
                
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)      
        
        if planes == 64:
            self.globalAvgPool = nn.AvgPool1d(1250, stride=1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool1d(625, stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool1d(313, stride=1)
        elif planes == 512:
            self.globalAvgPool = nn.AvgPool1d(157, stride=1)
            
        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
        self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)
        self.sigmoid = nn.Sigmoid()
             

    def forward(self, x):
        
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
                
        
        out = self.conv2(out)        
        out = self.bn2(out)        


        if self.downsample is not None:
            residual = self.downsample(x)
            
        original_out = out
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1),1)
        out = out * original_out
        out += residual
        out=self.relu(out)
        
        return out
    
class ECGNet(nn.Module):
    def __init__(self,block,layers, num_classes,num_external):
        super(ECGNet, self).__init__()
        self.inplanes = 64
        self.external = num_external
        self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7,bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)                   
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(.2)       
        self.fc = nn.Linear(512 * block.expansion+self.external, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self,x,x2): 

        x = self.conv1(x) 
        x = self.bn1(x)
        x = self.relu(x)  
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x3 = torch.cat([x,x2], dim=1)
        x3 = self.relu(x3)     
        x4 = self.fc(x3)                     
        return x4
    
    
    
# Load challenge data.
def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file,'r') as f:
        header_data=f.readlines()

    return data, header_data

# Find unique classes.
def get_classes(input_directory, filenames):
    classes = set()
    for filename in filenames:
        input_file=os.path.join(input_directory,filename)
        with open( input_file, 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    tmp = l.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())
    return sorted(classes)



def train(x_train,x_val,x_train_external,x_val_external,y_train,y_val):
    # model
    
    num_class=np.shape(y_val)[1]
    num_external=np.shape(x_val_external)[1]
    
    model = ECGNet(BasicBlock, [3, 4, 6, 3],num_classes= num_class,num_external=num_external)
    model = model.to(device)
    
    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    wc = y_train.sum(axis=0)
    wc = 1. / (np.log(wc)+1)

    w = torch.tensor(wc, dtype=torch.float).to(device)
#   criterion1 = utils.WeightedMultilabel(w)
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.BCEWithLogitsLoss()
    
    lr = config.lr
    start_epoch = 1
    stage = 1
    best_auc = -1
    
       
    # =========>开始训练<=========
    print("*" * 10, "step into stage %02d lr %.5f" % (stage, lr))
    for epoch in range(start_epoch, config.max_epoch + 1):
        since = time.time()
        train_loss,train_auc= train_epoch(model, optimizer, criterion1,x_train,x_train_external,y_train)
        val_loss,val_auc= val_epoch(model, criterion2, x_val,x_val_external,y_val)
        print('#epoch:%02d stage:%d train_loss:%.4f train_auc:%.4f  val_loss:%.4f val_auc:%.4f  time:%s'
              % (epoch, stage, train_loss, train_auc,val_loss,val_auc, utils.print_time_cost(since)))
            
        if epoch in config.stage_epoch:
            stage += 1
            lr /= config.lr_decay
            print("*" * 10, "step into stage %02d lr %.5f" % (stage, lr))
            utils.adjust_learning_rate(optimizer, lr)
    return model

def train_epoch(model, optimizer, criterion,x_train,x_train_external,y_train):
    model.train()
    auc_meter,loss_meter, it_count = 0, 0,0
    batch_size=config.batch_size

    for i in range(0,len(x_train)-batch_size,batch_size):      
        inputs1 = torch.tensor(x_train[i:i+batch_size],dtype=torch.float,device=device)
        inputs2 = torch.tensor(x_train_external[i:i+batch_size],dtype=torch.float,device=device)
        target =  torch.tensor(y_train[i:i+batch_size],dtype=torch.float,device=device)         
        output = model.forward(inputs1,inputs2) 
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        auc_meter = auc_meter+ utils.calc_auc(target, torch.sigmoid(output)) 
        
    return loss_meter / it_count, auc_meter/it_count

def val_epoch(model, criterion, x_val,x_val_external,y_val):
    model.eval()
    auc_meter,loss_meter, it_count = 0, 0,0
    batch_size=config.batch_size
    
    with torch.no_grad():
        for i in range(0,len(x_val)-batch_size,batch_size):      
            inputs1 = torch.tensor(x_val[i:i+batch_size],dtype=torch.float,device=device)
            inputs2 = torch.tensor(x_val_external[i:i+batch_size],dtype=torch.float,device=device)
            target =  torch.tensor(y_val[i:i+batch_size],dtype=torch.float,device=device)
            output = model(inputs1,inputs2)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1 
            auc_meter =auc_meter + utils.calc_auc(target, torch.sigmoid(output)) 
    return loss_meter / it_count, auc_meter/ it_count


def train_12ECG_classifier(input_directory, output_directory):
    
    input_files=[]
    header_files=[]
    
    train_directory=input_directory
    for f in os.listdir(train_directory):
        if os.path.isfile(os.path.join(train_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
            g = f.replace('.mat','.hea')
            input_files.append(f)
            header_files.append(g)

    # the 27 scored classes
    classes_weight=['270492004','164889003','164890007','426627000','713426002','445118002','39732003',
                    '164909002','251146004','10370003','164947007','111975006','47665007','427393009',
                    '426177001','426783006','427084000','59931005','164917005','164934002','698252002',
                    '713427006','284470004','427172004']
       
    classes_name=sorted(classes_weight)
    num_files=len(input_files)
    num_class=len(classes_name)
    normal_index=classes_name.index('426783006')
    
    
    def replace_equivalent_classes(classes):
        for i, x in enumerate(classes):
            if x=='59118001':
                classes[i]='713427006'
            if x=='63593006':
                classes[i]='284470004'
            if x=='17338001':
                classes[i]='427172004'
        return classes
            
    
    # initilize the array    
    set_length=5000
    data_num = np.zeros((num_files,12,set_length))   
    classes_num=np.zeros((num_files,num_class))
    data_external= np.zeros((num_files,2))
    
    for cnt,f in enumerate(input_files):
        classes=[]
        tmp_input_file = os.path.join(train_directory,f)
        data,header_data = load_challenge_data(tmp_input_file)  
               
        tmp_hea = header_data[0].split(' ')
        ptID = tmp_hea[0][0]                    
        num_leads = int(tmp_hea[1])
        sample_Fs= int(tmp_hea[2])
        tmp_length= int(tmp_hea[3])
        gain_lead = np.zeros(num_leads)
             
        if sample_Fs==1000:   
            rs_idx=range(0,len(data[0]),2)   
            data=data[:,rs_idx]         
                            
        for ii in range(num_leads):
            tmp_hea = header_data[ii+1].split(' ')
            gain_lead[ii] = int(tmp_hea[2].split('/')[0])            
        
        
        for i,lines in enumerate(header_data):
            if lines.startswith('#Age'):
                tmp_age = lines.split(': ')[1].strip()
                age = int(tmp_age if tmp_age != 'NaN' else 57)
                age=age/100 
            elif lines.startswith('#Sex'):
                tmp_sex = lines.split(': ')[1]
                if tmp_sex.strip()=='Female':
                    sex =1
                else:
                    sex=0

            elif lines.startswith('#Dx'):
                tmp = lines.split(': ')[1].split(',')
                for c in tmp:
                    classes.append(c.strip())                     
                classes = replace_equivalent_classes(classes)   
                
                    
                for j in classes:                         
                    if j in classes_name:
                        class_index=classes_name.index(j)
                        classes_num[cnt,class_index]=1    
                        
            
        
        data_external[cnt,0] =age 
        data_external[cnt,1] =sex


        if data.shape[1]>= set_length:
            data_num[cnt,:,:] = data[:,: set_length]*gain_lead[0]
        else:
            length=data.shape[1]
            data_num[cnt,:,:length] = data*gain_lead[0]         
                        
    #split the training set and testing set
    x_train,x_val,x_train_external,x_val_external,y_train,y_val = train_test_split(data_num,data_external,
                                               classes_num,test_size=0.01, random_state=2020)
    #build the pre_train model
    model= train(x_train,x_val,x_train_external,x_val_external,y_train,y_val)    
    
    #save the model
    output_directory=os.path.join(output_directory, 'resnet_0816.pkl')
    torch.save(model, output_directory)    
    