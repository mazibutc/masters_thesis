#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pyreadr
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


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


X_tst=X_faulty_test[X_faulty_test['simulationRun']==1]
for i in range(2,21):
    X_t=X_faulty_test[X_faulty_test['simulationRun']==i]
    X_tst=X_tst.append(X_t)


# In[ ]:


data=X_trn.drop(columns=['faultNumber','simulationRun','sample'])


# In[ ]:


fault_free_scaler=preprocessing.StandardScaler()
fault_free_scaler_dyn=preprocessing.StandardScaler()


# In[ ]:


X_scaled=fault_free_scaler.fit_transform(data)


# In[ ]:


def PCA_train(X_input,pc,alpha):
    pca=PCA(n_components=pc,svd_solver='full')
    T_scores=pca.fit_transform(X_input)
    cumulative_var_ratio=round(sum(pca.explained_variance_ratio_),2)
    X_predicted=pca.inverse_transform(T_scores)
    SPE_trn=np.sum(np.square(X_predicted-X_input),axis=1)
    Threshold=round(np.percentile(SPE_trn,alpha),2)
    return Threshold,cumulative_var_ratio


# In[ ]:


##20 is the number of simulation runs for the test dataset
def Test_vec():
    n_comp=20*960
    test_vec=np.ones(n_comp,dtype=int)
    test_vec=test_vec>0
    test_vec[0:160]=False ## 0 means fault_free
    for i in range(1,20):
        test_vec[(960)*i:(960)*i+160]=False
    return test_vec


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


n=20


# In[ ]:


def test_vector(time_steps):
    n_comp=n*(960-time_steps)
    test_vec_dyn=np.ones(n_comp,dtype=int)
    test_vec_dyn=test_vec_dyn>0
    test_vec_dyn[0:160-time_steps]=False ## 0 means fault_free
    for i in range(1,n):
        test_vec_dyn[(960-time_steps)*i:(960-time_steps)*i+160-time_steps]=False
    return test_vec_dyn
    


# In[ ]:


def PCA_pred(time_steps,alpha,static):
    anomaly=pd.DataFrame()
    b=np.zeros(2)
    FDR_array=np.zeros(2)
    if static=='yes':
        X=X_scaled
        test_vec=Test_vec()
        pc=15
    else:
        X=Expanded_matrix(data,time_steps,500)
        X=fault_free_scaler_dyn.fit_transform(X)
        test_vec=test_vector(time_steps)
        pc=85
    pca=PCA(n_components=pc,svd_solver='full')
    pca.fit_transform(X)
    Threshold,cum_var=PCA_train(X,pc,alpha)
    for i in range(1,21):
        X_test=X_tst[X_tst['faultNumber']==i]
        X_test=X_test.drop(columns=['sample','simulationRun','faultNumber'])
        if static=='yes':
            X_test_scaled=fault_free_scaler.transform(X_test)
            Z_scores=pca.transform(X_test_scaled)
        else:
            Xtest_dyn=Expanded_matrix(X_test,time_steps,simrun_examples=960)
            X_test_scaled=fault_free_scaler_dyn.transform(Xtest_dyn)
            Z_scores=pca.transform(X_test_scaled)
        Xtest_predicted=pca.inverse_transform(Z_scores)
        SPE_test=np.sum(np.square(Xtest_predicted-X_test_scaled),axis=1)
        anomaly['SPE_test']=SPE_test
        anomaly['Threshold']=Threshold
        anomaly['y_true']=test_vec
        anomaly['y_predicted']=anomaly['SPE_test']>anomaly['Threshold']
        tn, fp, fn, tp = confusion_matrix(anomaly['y_true'],anomaly['y_predicted']).ravel()
        fdr=tp/(tp+fn)*100
        far=fp/(fp+tn)*100
        b[0]=np.round(fdr,2)
        b[1]=np.round(far,2)
        FDR_array=np.vstack((FDR_array,b))
    FDR_tbl=pd.DataFrame(FDR_array)
    FDR_tbl.columns=['FDR','FAR']
    print(FDR_tbl)


# In[64]:


PCA_pred(alpha=99.3,static='no',time_steps=3)


# In[65]:


PCA_pred(alpha=99.5,static='no',time_steps=3)


# In[65]:


PCA_pred(alpha=99.3,static='yes',time_steps=3)


# In[ ]:




