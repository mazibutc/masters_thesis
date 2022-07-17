#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pyreadr
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix


# In[ ]:


from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector


# In[ ]:


X_faultfree_train=pyreadr.read_r("TEP_FaultFree_Training.RData")
X_faulty_train=pyreadr.read_r("TEP_Faulty_Training.RData")
X_faultfree_test=pyreadr.read_r("TEP_FaultFree_Testing.RData")
X_faulty_test=pyreadr.read_r("TEP_Faulty_Testing.RData")


# In[ ]:


X_faultfree_train=X_faultfree_train['fault_free_training']
X_faulty_train=X_faulty_train['faulty_training']
X_faultfree_test=X_faultfree_test['fault_free_testing']
X_faulty_test=X_faulty_test['faulty_testing']


# In[ ]:


X_trn=X_faultfree_train[X_faultfree_train['simulationRun']==1]
for i in range(2,152):
    X=X_faultfree_train[X_faultfree_train['simulationRun']==i]
    X_trn=X_trn.append(X)


# In[ ]:


X_test=X_faulty_test[X_faulty_test['simulationRun']==1]
for i in range(2,21):
    X_t=X_faulty_test[X_faulty_test['simulationRun']==i]
    X_test=X_test.append(X_t)


# In[ ]:


data=X_trn.drop(columns=['faultNumber','simulationRun','sample'])


# In[ ]:


scaler=preprocessing.StandardScaler()
data_scaled=scaler.fit_transform(data)


# In[ ]:


time_steps=3
lookback=time_steps
simulation_runs=int(X_trn.shape[0]/500) ## number of simulation runs


# In[ ]:


test_simulation_runs=20 ##int(X_test.shape[0]/960) ## for testing
time_step_number=test_simulation_runs*(960-lookback+1)


# In[ ]:


def sequence_input(X,time_steps):
    X=np.array(X)
    n=X.shape[0]
    x=[] #list
    for i in range(n-time_steps+1):
        x.append(X[i:i+time_steps])  ## x is a list of arrays
    return np.array(x)    #returns n dimensional array


# In[ ]:


def Input_transform(X_scaled,time_steps,simulation_runs):
    ## n is number of simulation runs
    X_array=np.empty((simulation_runs,500,52))
    for i in range(simulation_runs):
        X_array[i]=X_scaled[500*i:500*i+500,:] ### splits training dataset into individual simulation runs of 500 data points each
    a1=sequence_input(X_array[0],time_steps)
    for j in range(1,simulation_runs):
        X_sequence=sequence_input(X_array[j],time_steps)
        a2=np.vstack((a1,X_sequence))
        a1=a2
    return a1


# In[ ]:


## transforms test dataset
def Input_test_transform(X_scaled,time_steps,test_simulation_runs):
    ## n is number of simulation runs
    X_array=np.empty((test_simulation_runs,960,52))
    for i in range(test_simulation_runs):
        X_array[i]=X_scaled[960*i:960*i+960,:] ## each simulation run takes 960 data points
    a1=sequence_input(X_array[0],time_steps)
    for j in range(1,test_simulation_runs):
        X_sequence=sequence_input(X_array[j],time_steps)
        a2=np.vstack((a1,X_sequence))
        a1=a2
    return a1


# In[ ]:


## converts 3D array to 2D array
def flatten(X):
    X_2D=np.empty((X.shape[0],X.shape[2]))
    for i in range(X.shape[0]):
        X_2D[i]=X[i,X.shape[1]-1,:]
    return X_2D 


# In[ ]:


fault=np.ones(time_step_number,dtype=int)
fault=fault>0
fault[0:160-lookback+1]=False
for i in range(1,test_simulation_runs):
    fault[(960-lookback+1)*i:(960-lookback+1)*i+160-lookback+1]=False


# In[ ]:


Input_Xtrain=Input_transform(data_scaled,time_steps,simulation_runs)
n_features=Input_Xtrain.shape[2]


# In[ ]:


from keras.optimizers import Adam
opt=Adam(learning_rate=0.001)


# In[ ]:


for j in range(1,31):
    model=Sequential()
    model.add(LSTM(140,activation='elu',input_shape=(lookback,n_features),return_sequences=True))
    model.add(LSTM(120,activation='elu',return_sequences=False))
    model.add(RepeatVector(lookback))
    model.add(LSTM(120,activation='elu',return_sequences=True))
    model.add(LSTM(140,activation='elu',return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(optimizer=opt,loss='mse')
    model.fit(Input_Xtrain,Input_Xtrain,epochs=100,validation_split=0.1,batch_size=128,verbose=0)
    yhat=model.predict(Input_Xtrain,verbose=0)
    yhat=flatten(yhat)
    xtrain=flatten(Input_Xtrain)
    Q_statistic=np.sum(np.square(xtrain-yhat),axis=1)
    Thresh=round(np.percentile(Q_statistic,99.5),1)
    anomaly_aec=pd.DataFrame(columns=['Q_statistics','Threshold','y_true','y_predicted'])
    a=np.zeros(2)
    FDR_aec_array=np.zeros(2)
    for i in range(1,21):
        fault_data=X_test[X_test['faultNumber']==i]
        fault_data=fault_data.drop(columns=['faultNumber','simulationRun','sample'])
        scaled_data=scaler.transform(fault_data)
        Input_Xtest=Input_test_transform(scaled_data,time_steps,test_simulation_runs)
        yhat_test=model.predict(Input_Xtest,verbose=0)
        xtest=flatten(Input_Xtest)
        yhat_test=flatten(yhat_test)
        SPE=np.sum(np.square(xtest-yhat_test),axis=1)
        anomaly_aec['Q_statistics']=SPE
        anomaly_aec['Threshold']=Thresh
        anomaly_aec['y_true']=fault
        anomaly_aec['y_predicted']=SPE>Thresh
        tn, fp, fn, tp = confusion_matrix(anomaly_aec['y_true'],anomaly_aec['y_predicted']).ravel()
        fdr=tp/(tp+fn)*100
        far=fp/(fp+tn)*100
        a[0]=np.round(fdr,2)
        a[1]=np.round(far,2)
        FDR_aec_array=np.vstack((FDR_aec_array,a))
    FDR_aec_tbl=pd.DataFrame(FDR_aec_array)
    FDR_aec_tbl.columns=['FDR','FAR']
    print('######################################')
    print(FDR_aec_tbl)


# In[ ]:


pd.set_option("display.max_rows",None)


# In[ ]:


Loss_df=pd.DataFrame()
Loss_df['Validation']=model.history.history['val_loss']
Loss_df['Training']=model.history.history['loss']


# In[ ]:


print(Loss_df)

