#!/usr/bin/env python

import numpy as np, os, sys
import joblib
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_features(data,header_data): 
    set_length=5000
    data_num = np.zeros((1,12,set_length))
    data_external= np.zeros((1,2))
    
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
        if i==0:
            rs=lines.split(' ')[2]
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

    if data.shape[1]>= set_length:
        data_num[:,:,:] = data[:,: set_length]*gain_lead[0]
    else:
        length=data.shape[1]
        data_num[:,:,:length] = data*gain_lead[0]
        
    data_num= data_num.reshape(1,12,-1)   
    
    data_external[:,0] =age 
    data_external[:,1] =sex      
    
    return data_num,data_external

def load_12ECG_model(input_directory):
    # load the model from disk 
    f_out='resnet_0816.pkl'
    filename = os.path.join(input_directory,f_out)
    loaded_model = torch.load(filename,map_location=device)
    return loaded_model


def run_12ECG_classifier(data,header_data,model):       
    classes=['270492004','164889003','164890007','426627000','713426002','445118002','39732003',
             '164909002','251146004','10370003','164947007','111975006','47665007','427393009',
             '426177001','426783006','427084000','59931005','164917005','164934002','698252002',
             '713427006','284470004','427172004']
    
    classes=sorted(classes)
    num_classes = len(classes)
    
    # Use your classifier here to obtain a label and score for each class. 
    feats_reshape,feats_external = get_features(data,header_data)
    feats_reshape = torch.tensor(feats_reshape,dtype=torch.float,device=device)
    feats_external = torch.tensor(feats_external,dtype=torch.float,device=device)
    
    
    pred = model.forward(feats_reshape,feats_external)
    pred = torch.sigmoid(pred)
    
    current_score = pred.squeeze().cpu().detach().numpy()  
    
    current_label = np.zeros(24,)  
    
#     cutoff=[0.15, 0.15, 0.15 ,0.15, 0.15 ,0.15, 0.15, 0.15, 0.15,
#             0.15, 0.15, 0.15 ,0.15, 0.15 ,0.15, 0.15, 0.15, 0.15,
#             0.15, 0.15, 0.15, 0.15, 0.15, 0.35]   
    
#     for i in range(24):
#         if current_score[i]>cutoff[i]:
#             current_label[i]=1
#         else:
#             current_label[i]=0              
#     current_label=current_label.astype(int) 
    
    current_label=np.where(current_score>0.15,1,0)       
    num_positive_classes = np.sum(current_label)
    
    #窦性心律标签处于有评分的标签排序后的第14位
    normal_index=classes.index('426783006')
    max_index=np.argmax(current_score)               
       
    ##至少为一个标签，如果所有标签都没有，就将概率最大的设为1       
    if num_positive_classes==0:
        current_label[max_index]=1        
            
    return current_label, current_score, classes

