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
from keras.layers import Dense
from keras.models import Sequential


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


data=X_faultfree_train.drop(columns=['faultNumber','simulationRun','sample'])


# In[ ]:


scaler=preprocessing.StandardScaler()
data_scaled=scaler.fit_transform(data)


# In[ ]:


## num of principal components
l=15 


# In[ ]:


fault=np.ones(480000,dtype=int)


# In[ ]:


fault=fault>0


# In[ ]:


fault[0:160]=False
for i in range(1,500):
    fault[960*i:960*i+160]=False


# In[ ]:


for j in range(1,31):
    model=Sequential()
    model.add(Dense(100,activation='relu',kernel_initializer='glorot_uniform',input_shape=(data_scaled.shape[1],)))
    model.add(Dense(l,activation='relu',kernel_initializer='glorot_uniform'))
    model.add(Dense(100,activation='relu',kernel_initializer='glorot_uniform'))
    model.add(Dense(data_scaled.shape[1],activation='relu',kernel_initializer='glorot_uniform'))
    opt=keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(loss='mse',optimizer=opt)
    Num_Epochs=1000
    Batch_size=2000
    model_train=model.fit(data_scaled,data_scaled,batch_size=Batch_size,
                  epochs=Num_Epochs, validation_split= 0.05, verbose=0)
    Thresh=63
    anomaly_aec=pd.DataFrame(columns=['Q_statistics','Threshold','y_true','y_predicted'])
    a=np.zeros(2)
    FDR_aec_array=np.zeros(2)
    for i in range(1,21):
        fault_data=X_faulty_test[X_faulty_test['faultNumber']==i]
        fault_data=fault_data.drop(columns=['faultNumber','simulationRun','sample'])
        scaled_data=scaler.transform(fault_data)
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

