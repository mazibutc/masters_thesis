#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pyreadr
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix


# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


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


def exp_matrix(time_steps,X):
    X_mat=np.ones((len(X)-time_steps,X.shape[1]*time_steps+X.shape[1]))
    for i in range(len(X_mat)):
        for j in range(time_steps+1):
            X_mat[i,0+j*52:j*52+52]=np.array(X[i:time_steps+i+1])[time_steps-j,:]
    return X_mat


# In[ ]:


def Expanded_matrix(X,time_steps,simrun_examples):
    Xmat1=exp_matrix(time_steps,X[0:simrun_examples])
    for i in range(1,int(len(X)/simrun_examples)):
        Xmat2=exp_matrix(time_steps,X[simrun_examples*i:i*simrun_examples+simrun_examples])
        X_expn=np.vstack((Xmat1,Xmat2))
        Xmat1=X_expn
    return X_expn


# In[ ]:


time_steps=2
train_examples=500


# In[ ]:


X_trn=X_faultfree_train[X_faultfree_train['simulationRun']==1]
for i in range(2,152):
    X=X_faultfree_train[X_faultfree_train['simulationRun']==i]
    X_trn=X_trn.append(X)


# In[ ]:


X_test=X_faulty_test[X_faulty_test['simulationRun']==1]
for i in range(2,201):
    X_t=X_faulty_test[X_faulty_test['simulationRun']==i]
    X_test=X_test.append(X_t)


# In[ ]:


data=X_trn.drop(columns=['faultNumber','simulationRun','sample'])


# In[ ]:


data_dyn=Expanded_matrix(data,time_steps,train_examples)


# In[ ]:


scaler=preprocessing.StandardScaler()
data_scaled=scaler.fit_transform(data_dyn)


# In[ ]:


test_examples=960
n_comp=200*(960-time_steps)


# In[ ]:


simulation_runs=int(X_trn.shape[0]/500) ## number of simulation runs


# In[ ]:


test_simulation_runs=200
time_step_number=test_simulation_runs*(960-time_steps)


# In[ ]:


fault=np.ones(n_comp,dtype=int)


# In[ ]:


fault=fault>0


# In[ ]:


fault[0:160-time_steps]=False
for i in range(1,test_simulation_runs):
    fault[(960-time_steps)*i:(960-time_steps)*i+160-time_steps]=False


# In[ ]:


opt=Adam(learning_rate=0.001)


# In[ ]:


for j in range(1,31):
    model=Sequential()
    model.add(Dense(120,activation='tanh',kernel_initializer='glorot_uniform',input_shape=(data_scaled.shape[1],)))
    model.add(Dense(180,activation='linear',kernel_initializer='glorot_uniform'))
    model.add(Dense(120,activation='tanh',kernel_initializer='glorot_uniform'))
    model.add(Dense(data_scaled.shape[1],activation='linear',kernel_initializer='glorot_uniform'))
    model.compile(loss='mse',optimizer=opt)
    Num_Epochs=60
    Batch_size=128
    model_train=model.fit(data_scaled,data_scaled,batch_size=Batch_size,
                  epochs=Num_Epochs, validation_split= 0.1, verbose=0)
    yhat=model.predict(data_scaled,verbose=0)
    Q_statistic=np.sum(np.square(data_scaled-yhat),axis=1)
    Thresh=np.round(np.percentile(Q_statistic,99.5),1)
    anomaly_aec=pd.DataFrame(columns=['Q_statistics','Threshold','y_true','y_predicted'])
    a=np.zeros(2)
    FDR_aec_array=np.zeros(2)
    for i in range(1,21):
        fault_data=X_test[X_test['faultNumber']==i]
        fault_data=fault_data.drop(columns=['faultNumber','simulationRun','sample'])
        fault_data_dyn=Expanded_matrix(fault_data,time_steps,test_examples)
        scaled_data=scaler.transform(fault_data_dyn)
        scaled_pred=model.predict(scaled_data)
        SPE_aec=np.sum(np.square(scaled_data-scaled_pred),axis=1)
        anomaly_aec['Q_statistics']=SPE_aec
        anomaly_aec['Threshold']=Thresh
        anomaly_aec['y_true']=fault
        anomaly_aec['y_predicted']=SPE_aec>Thresh
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


Loss_df=pd.DataFrame()
Loss_df['Validation']=model.history.history['val_loss']
Loss_df['Training']=model.history.history['loss']


# In[ ]:


print(Loss_df)

