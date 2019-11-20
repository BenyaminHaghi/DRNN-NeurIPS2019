# Created by: Benyamin Haghi, California Institute of Technology (Caltech)
# benyamin.a.haghi@caltech.edu
# 26 Oct 2019
# CUDA GPU version

#############################################Import Libraries########################################
#from __future__ import print_function
import torch
import torch.distributions as tdist
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as data_utils
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pylab as pl
import torch.nn.init as init
import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr
import pandas
import math
import random
from math import sqrt
from numpy import concatenate
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
#from matplotlib import pyplot
import time
import h5py
from numba import jit, jitclass
#from joblib import Parallel, delayed
import multiprocessing
import json

device = torch.device('cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor)

#####################################################################################################################################################  

########################### Classes and Functions ########################### 

############### 1. Early Stopping
class EarlyStopping:
    """Early stops the training if validation loss dosen't improve after a given patience."""
    def __init__(self, patience=5, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.optim_model = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.optim_model = model
        elif score < self.best_score:
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.optim_model = self.save_checkpoint(val_loss, model)
            self.counter = 0
        final_model = self.optim_model
        return final_model
    
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        #if self.verbose:
            #print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), 'checkpoint.pt')
        optim_model = model
        self.val_loss_min = val_loss
        return optim_model

############### 2. convert series to supervised for supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

############### 4. DRNN
class DRNN_Network(nn.Module):
    def __init__(self,batch_size,n_features,n_neurons1,n_neurons2,n_outputs,n_steps,layer_key,flip_mode,dropout_coef):
        super(DRNN_Network, self).__init__()
        # Linear Model: linear(in_features,out_features), W: (out_features, in_features), input(Sth, in_features)
        # output(sth, out_features), b(out_features) , formua: y = x.A^T + b
        self.batch_size = batch_size
        self.n_features = n_features
        self.n_neurons1 = n_neurons1
        self.n_neurons2 = n_neurons2
        self.n_outputs = n_outputs
        self.n_steps = n_steps
        self.layer_key = layer_key
        self.flip_mode = flip_mode
        self.dropout_coef = dropout_coef
        self.Wx = nn.Linear(self.n_neurons1,self.n_neurons1,bias = False).cuda()
        self.Wr = nn.Linear(self.n_neurons1,self.n_neurons1,bias = False).cuda()
        self.Wu = nn.Linear(self.n_features,self.n_neurons1,bias = False).cuda()
        self.Wz = nn.Linear(self.n_outputs, self.n_neurons1,bias = False).cuda()
        self.tanh1 = nn.Tanh().cuda()
        self.dropout_layer = nn.Dropout2d(dropout_coef).cuda()
        self.bx = torch.nn.Parameter(torch.zeros(self.batch_size,self.n_neurons1)).cuda()
        if self.layer_key == 0:
            self.Wo = nn.Linear(self.n_neurons1,self.n_outputs).cuda()
        elif self.layer_key == 1: # 2 layers
            self.Whh = nn.Linear(self.n_neurons2,self.n_neurons2, bias = False).cuda()
            self.Whr = nn.Linear(self.n_neurons1,self.n_neurons2, bias = False).cuda()
            self.bh = torch.nn.Parameter(torch.zeros(self.batch_size,self.n_neurons2)).cuda()
            self.Wo = nn.Linear(self.n_neurons2,self.n_outputs).cuda()
        
    def init_hidden(self,): #x, r
        self.x = torch.FloatTensor(self.batch_size, self.n_neurons1).type(torch.FloatTensor)
        init.normal_(self.x, 0.0, 0.01)
        self.x =  Variable(self.x, requires_grad=True)
        self.r = self.tanh1(self.x)
        if self.layer_key == 1: 
            self.h = torch.zeros(self.batch_size, self.n_neurons2)
        
    def forward(self, u, z, epsilon, f):
        self.init_hidden()            
        u = u.permute(1, 0, 2) # seq_len * batch_size * n_features
        #print 'u.shape_internal: ', u.shape
        Tx = u.shape[0] # seq_len
        m = u.shape[1] # batch_size
        nx = u.shape[2] #n_features
        if self.dropout_coef > 0:
            u = self.dropout_layer(u)

        if (self.flip_mode == 1) and (epsilon >= 0): # sequence flip
            p = np.random.uniform(low=0, high=1, size=self.batch_size)
            p = np.where(p < epsilon, 0, 1)
            p = p.reshape((self.batch_size,1))
            p = torch.from_numpy(p)
        for l in range(Tx):
            if (self.flip_mode == 0) and (epsilon >= 0): 
                p = np.random.uniform(low=0, high=1, size=self.batch_size)
                p = np.where(p < epsilon, 0, 1)
                p = p.reshape((self.batch_size,1))
                p = torch.from_numpy(p)
            if epsilon >= 0:
                p = p.to(device)
                z = torch.where(p == 0, f, z)    
            self.x = self.x.cuda()
            self.r = self.r.cuda()
            u = u.cuda()
            z = z.cuda()

            self.x = self.Wx(self.x) + self.Wr(self.r) + self.Wu(u[l,:,:]) + self.Wz(z) + self.bx
            self.r = self.tanh1(self.x)
            if self.layer_key == 0:
                z = self.Wo(self.r)
            if self.layer_key == 1:
                self.h = self.tanh1(self.Whh(self.h) + self.Whr(self.r) + self.bh)
                z = self.Wo(self.h)
            z = torch.where(abs(z) > 1, self.tanh1(z), z)

        return z

#####################################################################################################################################################  

############### 5. Training DRNN

def DRNN_train(model,loss_fn,optimizer,epoch,train_dataloader,test_dataloader,f_test,layer_key,flip_mode,key,epsilon_e,epsilon_s,ep_flip_th):     
    
    epoch_train_loss_mat = []
    epoch_test_loss_mat = []
    epoch_cc_mat = []
    epoch_rmse_mat = []
    inv_y_mat = []
    inv_yhat_mat = []
    total_train_loss_mat = []
    total_test_loss_mat = []
    early_stopping = EarlyStopping(patience=5, verbose=False)
    
    for ep in range(epoch+1):
        print('ep: ', ep)
        
        if ep <= ep_flip_th:
            epsilon = ((epsilon_e[key] - epsilon_s[key])/ep_flip_th)*ep + epsilon_s[key]
        train_running_loss = 0.0
        test_running_loss = 0.0
        train_batch_counter = 0    
        test_batch_counter = 0
        model = model.to(device)
        model.train()
        for i, data in enumerate(train_dataloader):   
            optimizer.zero_grad()
            u, f = data
            u,f = u.to(device), f.to(device)
            if i == 0:
                z = f    
            z = model.forward(u, z, epsilon, f)
            loss = loss_fn(z,f) 
            loss.backward(retain_graph=True)
            optimizer.step()
            train_running_loss += loss.detach().item()   
            train_batch_counter += 1
        
        model.eval()   
        # Calculating Test Loss:
        epsilon = -1
        for i, data in enumerate(test_dataloader):
            u, f = data
            u,f = u.to(device), f.to(device)
            z = model.forward(u, z, epsilon, f)
            test_loss = loss_fn(z,f)
            test_running_loss += test_loss.detach().item()
            test_batch_counter += 1
            if i == 0:
                y_pred_hat = z
            else:
                y_pred_hat = torch.cat((y_pred_hat,z),0)
        
        epoch_train_loss_mat.append(train_running_loss/train_batch_counter)
        epoch_test_loss_mat.append(test_running_loss/test_batch_counter)
        print('Train loss: ', epoch_train_loss_mat[-1])
        print('Test loss: ', epoch_test_loss_mat[-1])
        # finding cc and rmse of each epoch    
        y_pred_hat = y_pred_hat.cpu()
        f_test = f_test.cpu()
        inv_yhat = y_pred_hat.detach().numpy()
        inv_y = f_test.numpy()
        for_range = min(inv_yhat.shape[0], inv_y.shape[0])
        inv_yhat = inv_yhat[0:for_range,:]
        inv_y = inv_y[0:for_range,:]
        yy = np.concatenate((inv_yhat,inv_y), axis = 1)
        yy = yy[~np.isnan(yy).any(axis=1)]
        inv_yhat = yy[:,0:n_kin]
        inv_y = yy[:,n_kin:]
        for_range = min(inv_yhat.shape[0], inv_y.shape[0])
        inv_y = inv_y[0:for_range ,:]
        inv_yhat = inv_yhat[0:for_range ,:]
        inv_y_mat.append(inv_y)
        inv_yhat_mat.append(inv_yhat)
        
        if for_range == 0:
            rmse_kin_mat = 100
            cc_kin_mat = -100
        else:
            cc_kin_mat = []
            rmse_kin_mat = []
            for j in range(0,n_kin):
                rmse = sqrt(mean_squared_error(inv_y[:,j], inv_yhat[:,j]))
                cc,p = pearsonr(inv_y[:,j],inv_yhat[:,j]) 
                rmse_kin_mat.append(rmse)
                cc_kin_mat.append(cc)
                
        epoch_rmse_mat.append(rmse_kin_mat)
        epoch_cc_mat.append(cc_kin_mat)
        
        #if param_reg_flag == 0: # we are in parameter selection mode, not in regression mode
        if ep > 0:
            optim_model_prev = optim_model
            
        optim_model = early_stopping(test_running_loss/test_batch_counter, model)
    
        if early_stopping.early_stop:
            break
        
        if math.isnan(test_running_loss/test_batch_counter) == True:
            if ep > 0:
                optim_model = optim_model_prev
            break
        
    #if param_reg_flag == 0: # we are in parameter selection mode, not in regression mode      
    #model.load_state_dict(torch.load('checkpoint.pt'))
    model = optim_model
    final_epoch = epoch_test_loss_mat.index(min(epoch_test_loss_mat))
    final_rmse = epoch_rmse_mat[final_epoch]
    final_cc = epoch_cc_mat[final_epoch]
    inv_y = inv_y_mat[final_epoch]
    inv_yhat = inv_yhat_mat[final_epoch]

    return model,final_epoch,final_rmse,final_cc,inv_y,inv_yhat

#####################################################################################################################################################

############### 6. Testing DRNN

def DRNN_test(model,test_dataloader,f_test): 
    model = model.to(device)
    model.eval()   
    # Calculating Test Loss:
    epsilon = -1
    for i, data in enumerate(test_dataloader):
        u, f = data
        u, f = u.to(device), f.to(device)
        if i == 0:
            z = f   
        z = model.forward(u, z, epsilon, f)
        if i == 0:
            y_pred_hat = z
        else:
            y_pred_hat = torch.cat((y_pred_hat,z),0)
    
    
    y_pred_hat = y_pred_hat.cpu()
    f_test = f_test.cpu()
    inv_yhat = y_pred_hat.detach().numpy()
    inv_y = f_test.numpy()
    for_range = min(inv_yhat.shape[0], inv_y.shape[0])
    inv_yhat = inv_yhat[0:for_range,:]
    inv_y = inv_y[0:for_range,:]
    yy = np.concatenate((inv_yhat,inv_y), axis = 1)
    yy = yy[~np.isnan(yy).any(axis=1)]
    inv_yhat = yy[:,0:n_kin]
    inv_y = yy[:,n_kin:]
    for_range = min(inv_yhat.shape[0], inv_y.shape[0])
        
    cc_kin_mat = []
    rmse_kin_mat = []
    if for_range == 0:
        rmse_kin_mat.append(100)
        cc_kin_mat.append(-100)
    else:
        for j in range(0,n_kin):
            rmse = sqrt(mean_squared_error(inv_y[0:for_range,j], inv_yhat[0:for_range,j]))
            cc,p = pearsonr(inv_y[0:for_range,j],inv_yhat[0:for_range,j]) 
            rmse_kin_mat.append(rmse)
            cc_kin_mat.append(cc)

    return cc_kin_mat,rmse_kin_mat,inv_y,inv_yhat

#####################################################################################################################################################  

############### 7. Cross validation for best parameters selection

def parameter_selection_block(param_in,n_datasets,epoch,u,f,cc_p,rmse_p,final_epoch_p,loss_fn_mat,epsilon_e,epsilon_s,ep_flip_th,valid_range_all,test_range_all):
    cc_valid = [] # for each dataset
    rmse_valid = []
    layer_key = param_in[0]
    flip_mode = param_in[1]
    key = param_in[2]
    optimizer_key = param_in[3]
    loss_fn_key = param_in[4]
    batch_size = param_in[5]
    n_hours = param_in[6]
    pca_coef = param_in[7]
    dropout_coef = param_in[8]
    regul_coef = param_in[9]
    lr = param_in[10]
    pca_shift_key = param_in[11]
    n_neurons1 = param_in[12]
    my_model = PCA(n_components=pca_coef, svd_solver='full')
    if layer_key == 0:
        n_neurons2 = 0
    elif layer_key == 1:
        n_neurons2 = param_in[13]
    if pca_shift_key == 0: # first pca, then shift
        n_steps = n_hours + 1
        u = my_model.fit_transform(u)
        n_neural_data = len(u.T)
        n_features = n_neural_data
        # shifting
        if n_hours != 0:
            reframed = series_to_supervised(u, n_hours, 1)
            u = reframed.values
    elif pca_shift_key == 1: # first shift, then pca
        n_steps = 1
        # shifting
        if n_hours != 0:
            reframed = series_to_supervised(u, n_hours, 1)
            u = reframed.values
        u = my_model.fit_transform(u)
        n_neural_data = len(u.T)
        n_features = n_neural_data

    f = f[n_hours:,:]
    u = u.reshape((u.shape[0],n_steps, n_neural_data)) # N*Tx*nx
    
    # valid, test, and train data generation
    final_epoch_mat = []
    loss_fn = loss_fn_mat[loss_fn_key]

    for jj in range(n_datasets):
        valid_start_index = int(math.ceil(valid_range_all[jj][0]*len(u)))
        valid_stop_index = int(math.ceil(valid_range_all[jj][1]*len(u)))
        #test_start_index = int(math.ceil(test_range_all[jj][0]*len(u)))
        #test_stop_index = int(math.ceil(test_range_all[jj][1]*len(u))) 
        u_valid = u[valid_start_index:valid_stop_index,:]
        f_valid = f[valid_start_index:valid_stop_index,:]
        #delete_index = range(valid_start_index,valid_stop_index) + range(test_start_index,test_stop_index)
        delete_index = range(valid_start_index,valid_stop_index)
        u_train = np.delete(u,delete_index,axis = 0)
        f_train = np.delete(f,delete_index,axis = 0)
        u_train = Variable(torch.Tensor(u_train).type(torch.FloatTensor), requires_grad=False)
        f_train = Variable(torch.Tensor(f_train).type(torch.FloatTensor), requires_grad=False)
        u_valid = Variable(torch.Tensor(u_valid).type(torch.FloatTensor), requires_grad=False)
        f_valid = Variable(torch.Tensor(f_valid).type(torch.FloatTensor), requires_grad=False)
        train_loader = data_utils.TensorDataset(u_train, f_train)
        train_dataloader = data_utils.DataLoader(train_loader, batch_size = batch_size, shuffle = False, num_workers = 2, drop_last=True)
        valid_loader = data_utils.TensorDataset(u_valid, f_valid)
        valid_dataloader = data_utils.DataLoader(valid_loader, batch_size = batch_size, shuffle = False, num_workers = 2, drop_last=True)
            
        # training and validation
        # train
        model = DRNN_Network(batch_size,n_features,n_neurons1,n_neurons2,n_outputs,n_steps,layer_key,flip_mode,dropout_coef)
        if optimizer_key == 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = regul_coef)
        elif optimizer_key == 1:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay = regul_coef)       
        model,final_epoch,final_rmse,final_cc,inv_y,inv_yhat = DRNN_train(model,loss_fn,optimizer,epoch,train_dataloader,valid_dataloader,f_valid,layer_key,flip_mode,key,epsilon_e,epsilon_s,ep_flip_th)
        rmse_valid.append(np.asscalar(np.mean(np.asarray(final_rmse))))
        cc_valid.append(np.asscalar(np.mean(np.asarray(final_cc))))
        final_epoch_mat.append(final_epoch)
    
    rmse_valid = np.asarray(rmse_valid)
    rmse_p.append(np.mean(rmse_valid))
    cc_valid = np.asarray(cc_valid)
    cc_p.append(np.mean(cc_valid))
    final_epoch_p.append(final_epoch_mat)
    
    return rmse_p, cc_p, final_epoch_p

#####################################################################################################################################################  

############### 8. Going through different combination of parameters - random search

def parameter_selection(u,f,valid_range_all,test_range_all,param_mat,epoch,loss_fn_mat,epsilon_e,epsilon_s,ep_flip_th):
    n_datasets = len(valid_range_all)

    cc_p = []  
    rmse_p = []
    final_epoch_p = []
    random_search_iter = 20
    random_search_mat = np.random.randint(0,len(param_mat), size=random_search_iter)
    print('Total number of Parameter Selection Iteration: ', len(param_mat))
    #for ii in range(len(param_mat)):
    for ii in random_search_mat:
    #for ii in range(1):
        param_in = param_mat[ii]
        print('Parameter Set: ', ii)
        rmse_p, cc_p, final_epoch_p = parameter_selection_block(param_in,n_datasets,epoch,u,f,cc_p,rmse_p,final_epoch_p,loss_fn_mat,epsilon_e,epsilon_s,ep_flip_th,valid_range_all,test_range_all)
    
    print('Parameter Selection Finished.')
    final_index = cc_p.index(max(cc_p))    
    final_params = param_mat[final_index]
    final_epoch = final_epoch_p[final_index]

    #return rmse_p, cc_p, final_epoch_p, final_params, final_epoch    # this is for grid search
    return final_params, final_epoch # for random search, we don't know the order

#####################################################################################################################################################  

############### 9. Generate training data

def Training_Data_Generation(u,f,test_range_all,final_params,pca_model,pca_yes_no):
    n_datasets = len(test_range_all)
    pca_shift_key = final_params[11]
    n_hours = final_params[6]
    if pca_shift_key == 0: # first pca, then shift
        n_steps = n_hours + 1
        if pca_yes_no == 1:
            u = pca_model.fit_transform(u)
        n_neural_data = len(u.T)
        n_features = n_neural_data
        # shifting
        if n_hours != 0:
            reframed = series_to_supervised(u, n_hours, 1)
            u = reframed.values           
    elif pca_shift_key == 1: # first shift, then pca
        n_steps = 1
        # shifting
        if n_hours != 0:
            reframed = series_to_supervised(u, n_hours, 1)
            u_train = reframed.values   
        if pca_yes_no == 1:
            u = pca_model.fit_transform(u)      
        n_neural_data = len(u.T)
        n_features = n_neural_data 
        
    f = f[n_hours:,:]
    
    for jj in range(n_datasets):
        test_start_index = int(math.ceil(test_range_all[jj][0]*len(u)))
        test_stop_index = int(math.ceil(test_range_all[jj][1]*len(u))) 
        u_test = u[test_start_index:test_stop_index,:]
        f_test = f[test_start_index:test_stop_index,:] 
        delete_index = range(test_start_index,test_stop_index)
        u_train = np.delete(u,delete_index,axis = 0)
        f_train = np.delete(f,delete_index,axis = 0)       
        
    return u_train, f_train, u_test, f_test, n_steps, n_neural_data, n_features

#####################################################################################################################################################

############### 10. Training with cross validation

def Training(u_train,f_train,u_test,f_test,test_range_all, final_params,final_epoch,loss_fn_mat,epsilon_e,epsilon_s,ep_flip_th,epoch,n_steps,n_neural_data,n_features):
    print('Training is in progress...')
    n_features = n_neural_data
    n_datasets = len(test_range_all)
    layer_key = final_params[0]
    flip_mode = final_params[1]
    key = final_params[2]
    optimizer_key = final_params[3]
    loss_fn_key = final_params[4]
    batch_size = final_params[5]
    n_hours = final_params[6]
    pca_coef = final_params[7]
    dropout_coef = final_params[8]
    regul_coef = final_params[9]
    lr = final_params[10]
    pca_shift_key = final_params[11]
    n_neurons1 = final_params[12]
    my_model = PCA(n_components=pca_coef, svd_solver='full')
    if layer_key == 0:
        n_neurons2 = 0
    elif layer_key == 1:
        n_neurons2 = final_params[13]

    u_train = u_train.reshape((u_train.shape[0],n_steps, n_neural_data)) # N*Tx*nx
    u_test = u_test.reshape((u_test.shape[0],n_steps, n_neural_data)) # N*Tx*nx    
    
    cc_test = []
    rmse_test = [] 
    loss_fn = loss_fn_mat[loss_fn_key]
    for jj in range(n_datasets):  
        u_train = Variable(torch.Tensor(u_train).type(torch.FloatTensor), requires_grad=False)
        f_train = Variable(torch.Tensor(f_train).type(torch.FloatTensor), requires_grad=False)
        u_test = Variable(torch.Tensor(u_test).type(torch.FloatTensor), requires_grad=False)
        f_test = Variable(torch.Tensor(f_test).type(torch.FloatTensor), requires_grad=False)        
        train_loader = data_utils.TensorDataset(u_train, f_train)
        train_dataloader = data_utils.DataLoader(train_loader, batch_size = batch_size, shuffle = False, num_workers = 2, drop_last=True)
        test_loader = data_utils.TensorDataset(u_test, f_test)
        test_dataloader = data_utils.DataLoader(test_loader, batch_size = batch_size, shuffle = False, num_workers = 2, drop_last=True)
        
        # Train and inference
        # Train
        model = DRNN_Network(batch_size,n_features,n_neurons1,n_neurons2,n_outputs,n_steps,layer_key,flip_mode,dropout_coef)
        if optimizer_key == 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = regul_coef)
        elif optimizer_key == 1:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay = regul_coef)
        model,final_epoch,final_rmse,final_cc,inv_y,inv_yhat = DRNN_train(model,loss_fn,optimizer,epoch,train_dataloader,test_dataloader,f_test,layer_key,flip_mode,key,epsilon_e,epsilon_s,ep_flip_th)
        rmse_test.append(np.asscalar(np.mean(np.asarray(final_rmse))))
        cc_test.append(np.asscalar(np.mean(np.asarray(final_cc))))
        if np.asscalar(np.mean(np.asarray(final_cc))) == max(cc_test):
            test_dataloader_best = test_dataloader
            f_test_best = f_test
            best_model = model
            inv_y_best = inv_y
            inv_yhat_best = inv_yhat
    
    rmse_test = np.asarray(rmse_test)
    final_rmse = np.mean(rmse_test)
    cc_test = np.asarray(cc_test)
    final_cc = np.mean(cc_test)
    
    return final_rmse, final_cc, test_dataloader_best, f_test_best, best_model, inv_y_best, inv_yhat_best

#####################################################################################################################################################  

############### 11. Generate test data

def Testing_Data_Generation(u,f,train_range,test_range,final_params,pca_model,pca_yes_no):
    n_datasets = len(test_range)
    pca_shift_key = final_params[11]
    n_hours = final_params[6]
    if pca_shift_key == 0: # first pca, then shift
        n_steps = n_hours + 1
        if pca_yes_no == 1:
            u = pca_model.fit_transform(u)
        n_neural_data = len(u.T)
        n_features = n_neural_data
        # shifting
        if n_hours != 0:
            reframed = series_to_supervised(u, n_hours, 1)
            u = reframed.values           
    elif pca_shift_key == 1: # first shift, then pca
        n_steps = 1
        # shifting
        if n_hours != 0:
            reframed = series_to_supervised(u, n_hours, 1)
            u_train = reframed.values   
        if pca_yes_no == 1:
            u = pca_model.fit_transform(u)      
        n_neural_data = len(u.T)
        n_features = n_neural_data 
        
    f = f[n_hours:,:]
    
    for jj in range(n_datasets):
        train_start_index = 0
        train_stop_index = int(math.ceil(train_range[jj][1]*len(u)))
        test_start_index = int(math.ceil(test_range[jj][0]*len(u)))
        test_stop_index = int(math.ceil(test_range[jj][1]*len(u))) 
        u_test = u[test_start_index:test_stop_index,:]
        f_test = f[test_start_index:test_stop_index,:] 
        u_train = u[train_start_index:train_stop_index,:]
        f_train = f[train_start_index:train_stop_index,:]       
        
    return u_train, f_train, u_test, f_test, n_steps, n_neural_data, n_features


#####################################################################################################################################################  

############### 10. Testing

def Testing(u_test,f_test, final_params,final_epoch,loss_fn_mat,epsilon_e,epsilon_s,ep_flip_th,epoch,pca_model,pca_yes_no):
    n_datasets = len(test_range_all)
    layer_key = final_params[0]
    flip_mode = final_params[1]
    key = final_params[2]
    optimizer_key = final_params[3]
    loss_fn_key = final_params[4]
    batch_size = final_params[5]
    n_hours = final_params[6]
    pca_coef = final_params[7]
    dropout_coef = final_params[8]
    regul_coef = final_params[9]
    lr = final_params[10]
    pca_shift_key = final_params[11]
    n_neurons1 = final_params[12]
    my_model = PCA(n_components=pca_coef, svd_solver='full')
    if layer_key == 0:
        n_neurons2 = 0
    elif layer_key == 1:
        n_neurons2 = final_params[13]
    u_test = u_test.reshape((u_test.shape[0],n_steps, n_neural_data)) # N*Tx*nx
    u_test = Variable(torch.Tensor(u_test).type(torch.FloatTensor), requires_grad=False)
    f_test = Variable(torch.Tensor(f_test).type(torch.FloatTensor), requires_grad=False)    
    test_loader = data_utils.TensorDataset(u_test, f_test)
    test_dataloader = data_utils.DataLoader(test_loader, batch_size = batch_size, shuffle = False, num_workers = 2, drop_last=True)
        
    cc_kin_mat,rmse_kin_mat,inv_y,inv_yhat = DRNN_test(best_model,test_dataloader,f_test)   
        
    return cc_kin_mat,rmse_kin_mat,inv_y,inv_yhat
#####################################################################################################################################################  

########################### Main ########################### 

########################### 1. Parameters Generations

param_mat = []

layer_key_mat = [0] #0: one layer, 1: two layers
flip_mode_mat = [1] # once per token or once per sequence
key_mat = [2] #key0:Always sampling,key1:Sampling1,key2:Sampling2,key3:sampling3,key4:always groundtruth
optimizer_key_mat = [0] #0: Adam, 1: RMSprop
loss_func_key_mat = [0] #0:MSE, 1:SmoothL1Loss
batch_size_mat = [64]
n_hours_mat = [0]
pca_coef_mat = [0.4]
dropout_coef_mat = [0]
regul_coef_mat = [0]
lr_mat = [0.001]
pca_shift_key_mat = [0] #0: First PCA, then shift, 1: First shift, then PCA
n_neurons_mat = [25]

for i0 in range(len(layer_key_mat)):
    for i1 in range(len(flip_mode_mat)):
        for i2 in range(len(key_mat)):
            for i3 in range(len(optimizer_key_mat)):
                for i4 in range(len(loss_func_key_mat)):
                    for i5 in range(len(batch_size_mat)):
                            for i6 in range(len(n_hours_mat)):
                                for i7 in range(len(pca_coef_mat)):
                                    for i8 in range(len(dropout_coef_mat)):
                                        for i9 in range(len(regul_coef_mat)):
                                            for i10 in range(len(lr_mat)):
                                                for i11 in range(len(pca_shift_key_mat)):
                                                    for i12 in range(len(n_neurons_mat)):
                                                        if (i11 == 0) or (pca_shift_key_mat[i11] == 1 and flip_mode_mat[i1] == 0):
                                                            if layer_key_mat[i0] == 0:
                                                                param_mat.append([layer_key_mat[i0],flip_mode_mat[i1],key_mat[i2],optimizer_key_mat[i3],loss_func_key_mat[i4],batch_size_mat[i5],n_hours_mat[i6],pca_coef_mat[i7],dropout_coef_mat[i8],regul_coef_mat[i9],lr_mat[i10],pca_shift_key_mat[i11],n_neurons_mat[i12]])
                                                            elif layer_key_mat[i0] == 1:
                                                                for i13 in range(len(n_neurons_mat)):
                                                                    print 'i13: ', i13
                                                                    param_mat.append([layer_key_mat[i0],flip_mode_mat[i1],key_mat[i2],optimizer_key_mat[i3],loss_func_key_mat[i4],batch_size_mat[i5],n_hours_mat[i6],pca_coef_mat[i7],dropout_coef_mat[i8],regul_coef_mat[i9],lr_mat[i10],pca_shift_key_mat[i11],n_neurons_mat[i12],n_neurons_mat[i13]])
print param_mat
    
########################### 2. Initialization phase
#Parameters
n_kin = 4 
n_outputs = n_kin
epoch = 10
ep_flip_th = 9
loss_fn0 = torch.nn.MSELoss()
loss_fn1 = torch.nn.SmoothL1Loss() 
loss_fn_mat = [loss_fn0,loss_fn1]
epsilon_s = np.asarray([0, 0.25, 0.5, 0.9, 1])
epsilon_e = np.asarray([0, 0, 0, 0.5, 1])


#We have 10 cross-validation folds. In each fold, 10% of the data is a test set, 10% is a validation set, and 80% is the training set. So in the first fold, for example, 0-10% is validation, 10-20% is testing, and 20-100% is training.

valid_range_all=[[0,.1],[.1,.2],[.2,.3],[.3,.4],[.4,.5],

                 [.5,.6],[.6,.7],[.7,.8],[.8,.9],[.9,1]]

test_range_all=[[.1,.2],[.2,.3],[.3,.4],[.4,.5],[.5,.6],

                 [.6,.7],[.7,.8],[.8,.9],[.9,1],[0,.1]]

#Note that the training set is not aways contiguous. For example, in the second fold, the training set has 0-10% and 30-100%.

#In that example, we enter of list of lists: [[0,.1],[.3,1]]

train_range_all=[[[.2,1]],[[0,.1],[.3,1]],[[0,.2],[.4,1]],[[0,.3],[.5,1]],[[0,.4],[.6,1]],

                   [[0,.5],[.7,1]],[[0,.6],[.8,1]],[[0,.7],[.9,1]],[[0,.8]],[[.1,.9]]]

#####################################################################################################################################################  

wt_level = 1
day_final_params = []
day_final_epoch = []
day_final_rmse = []
day_final_cc = []
day_best_model = []
day_inv_y = []
day_inv_yhat = []
day_final_time = []
final_params = [0,1,1,0,0,64,10,50,0,0,0.001,0,25]
pca_model = PCA(n_components=final_params[7], svd_solver='full')
pca_yes_no = 0
final_epoch = 30
start = time.time()

#####################################################################################################################################################  

########################### 3. Data Loading

data_dir = './my_data/'
matlab_data = sio.loadmat(data_dir+'Neural_features_targets.mat')
neural_data = matlab_data['y']
targets = matlab_data['x_valid']
dataset = np.concatenate((targets,neural_data),axis = 1)
dataset = dataset.astype('float32')
u = dataset[:,n_kin:]
f = dataset[:,0:n_kin]

########################### 4. Processing Phase
u_train, f_train, u_val, f_val, n_steps, n_neural_data, n_features = Training_Data_Generation(u,f,test_range_all,final_params,pca_model,pca_yes_no)
print u_train.shape
print n_neural_data
final_rmse, final_cc, test_dataloader_best, f_test_best, best_model, inv_y, inv_yhat = Training(u_train,f_train,u_val,f_val,test_range_all, final_params,final_epoch,loss_fn_mat,epsilon_e,epsilon_s,ep_flip_th,epoch,n_steps,n_neural_data, n_features)

u_train, f_train, u_test, f_test, n_steps, n_neural_data, n_features = Testing_Data_Generation(u,f,[[0,0.8]],[[.8,1]],final_params,pca_model,pca_yes_no)
cc_kin_mat,rmse_kin_mat,inv_y,inv_yhat = Testing(u_test,f_test, final_params,final_epoch,loss_fn_mat,epsilon_e,epsilon_s,ep_flip_th,epoch,pca_model,pca_yes_no)    

stop = time.time()
print('time: ', stop - start)



########################### 4. Termination Phase - Plot

from matplotlib import pyplot
for j in range(0,n_kin):
    # Train data - calculate RMSE & correlation coefficient
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    cc,p = pearsonr(inv_y,inv_yhat)
    print("Test Pearson Correlation Coefficient: ", cc)   

    # plot some of training data
    pyplot.plot(inv_y[0:1000,j],'-b',label='Correct Value',linewidth=2.0)
    pyplot.plot(inv_yhat[0:1000,j],'-r',label='Predicted Value',linewidth=2.0)
    pyplot.xlabel('time')
    pyplot.ylabel('cm')
    pyplot.legend(loc='upper left')
    pyplot.show()
