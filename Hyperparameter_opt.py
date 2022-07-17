#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow import keras
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPooling2D, UpSampling2D, RepeatVector
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, Dense
from tensorflow.keras import optimizers


# In[ ]:


from kerastuner.tuners import RandomSearch
from kerastuner.engine.hypermodel import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA
import random


# In[ ]:


X_trn=X_faultfree_train[X_faultfree_train['simulationRun']==1]
for i in range(2,500):
    X=X_faultfree_train[X_faultfree_train['simulationRun']==i]
    X_trn=X_trn.append(X)


# In[ ]:


X_test=X_faulty_test[X_faulty_test['simulationRun']==1]
for i in range(2,500):
    X_t=X_faulty_test[X_faulty_test['simulationRun']==i]
    X_test=X_test.append(X_t)


# In[ ]:


X_trn=X_trn.drop(columns=['faultNumber','simulationRun','sample'])


# In[ ]:


time_steps=2


# In[ ]:


def exp_matrix(time_steps,X):
    X_mat=np.ones((len(X)-time_steps,X.shape[1]*time_steps+X.shape[1]))
    for i in range(len(X_mat)):
        for j in range(time_steps+1):
            X_mat[i,0+j*52:j*52+52]=np.array(X[i:time_steps+i+1])[time_steps-j,:]
    return X_mat
    


# In[ ]:


simrun_examples=500
def Expanded_matrix(X,time_steps,simrun_examples):
    Xmat1=exp_matrix(time_steps,X[0:simrun_examples])
    for i in range(1,int(len(X)/simrun_examples)):
        Xmat2=exp_matrix(time_steps,X[simrun_examples*i:i*simrun_examples+simrun_examples])
        X_expn=np.vstack((Xmat1,Xmat2))
        Xmat1=X_expn
    return X_expn


# In[ ]:


X_expanded=Expanded_matrix(X_trn,time_steps,simrun_examples)


# In[ ]:


fault_free_scaler=preprocessing.StandardScaler()


# In[ ]:


X_dyn_scaled=fault_free_scaler.fit_transform(X_expanded)


# In[ ]:


opt=optimizers.Adam(learning_rate=0.001)


# In[ ]:


def build_model(hp):
    func_activations=hp.Choice('activation_funcs',values=['linear','sigmoid','relu','tanh','elu'])
    mdl=Sequential()
    mdl.add(Dense(units=hp.Int("layer1",min_value=20,max_value=200,step=20),activation=hp.Choice('activation_func1',
                                                                                                 values=['linear','sigmoid','relu','tanh','elu']),
                  kernel_initializer='glorot_uniform',input_shape=(X_dyn_scaled.shape[1],)))
    mdl.add(Dense(units=hp.Int("layer2",min_value=20,max_value=200,step=20),activation=hp.Choice('activation_func2',
                                                                                                 values=['linear','sigmoid','relu','tanh','elu']),
                  kernel_initializer='glorot_uniform'))
    mdl.add(Dense(units=hp.Int("layer3",min_value=20,max_value=200,step=20),activation=hp.Choice('activation_func3',
                                                                                                 values=['linear','sigmoid','relu','tanh','elu']),
                  kernel_initializer='glorot_uniform'))
    mdl.add(Dense(X_dyn_scaled.shape[1],activation=hp.Choice('activation_func4',
                                                             values=['linear','sigmoid','relu','tanh','elu']),
                  kernel_initializer='glorot_uniform'))
    mdl.compile(loss='mse',optimizer=opt)
    return mdl


# In[ ]:


tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=50,
    directory='test_dir')


# In[ ]:


tuner.search(x=X_dyn_scaled, y=X_dyn_scaled,batch_size=280,epochs=100, validation_split=0.1)


# In[ ]:


bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]


# In[ ]:


print("[INFO] optimal number of filters in layer_1 layer: {}".format(bestHP.get("layer1")))
print("[INFO] optimal number of filters in layer_2 layer: {}".format(bestHP.get("layer2")))
print("[INFO] optimal number of filters in layer_3 layer: {}".format(bestHP.get("layer3")))


# In[ ]:


print("[INFO] optimal activation function: {}".format(bestHP.get("activation_func1")))
print("[INFO] optimal activation function: {}".format(bestHP.get("activation_func2")))
print("[INFO] optimal activation function: {}".format(bestHP.get("activation_func3")))
print("[INFO] optimal activation function: {}".format(bestHP.get("activation_func4")))

