# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from pomegranate import *
import copy
import re
import pywt
import networkx as nx
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

debug = 0
IDs = pd.read_csv(r"out\id-2.csv",header=None)
ECG = pd.read_csv(r"out\ECG-2.csv",header=None)
ECGLabel = pd.read_csv(r"out\ECGLabel-2.csv",header=None)
ECG = ECG.drop([248,249],axis=1)
ECGLabel = ECGLabel.drop([248,249],axis=1)
states = ['B','P','X','R','T']
#states = ['B','P','R','T']

def seperateBaselines(df):
    df2 = pd.DataFrame(columns=df.columns)
    for i in range(0,len(df)):
        st = ''.join(list(np.array(df.loc[[i]])[0]))
        st1 = re.split('(P)',st)
#        print(st)
        st1 = [re.split('(R)',x) for x in st1]
        result = ''
        for x in st1:
            if '' in x:
                x[0] = x[0].replace('B','X')
                y = ''.join(x)    
                result = result + y
            else:
                result = result + x[0]
        result = list(result)
        df2.loc[i] = result
    return df2
def getPCA(df):
    df2 = pd.DataFrame(columns=df.columns)
    for i in range(0,len(df)):
#        import pdb; pdb.set_trace()
        signal = np.array(df.loc[[i]])[0]
        signalGrad = np.gradient(signal)
        signalGrad2 = np.gradient(signalGrad)
        pca = PCA(n_components=1)
        principalComponents = pca.fit_transform(np.transpose(np.array([signal,signalGrad])))    
        df2.loc[i] = np.transpose(principalComponents)[0]
    return df2
def getInitialProb(df,states):
    initialProb = []
    for state in states:
        initialProb.append(np.sum((df.get_values()==state)*1)/df.size)
    return np.array(initialProb)
def getTransitionProb(df,states):
    num_states = len(states)
    roll_rates = pd.DataFrame(np.zeros([num_states, 
    num_states]),columns=states,index=states)    
    df_trans = pd.DataFrame(columns=["mo1","mo2"])
    for c1 in range(len(df.columns) - 1):
        c2 = c1 + 1
        trans = pd.concat([df[c1],df[c2]],axis=1)
        trans.columns = ["mo1","mo2"]
        df_trans = pd.concat([df_trans, trans],ignore_index=True)
    for s1 in states:
        for s2 in states:
            num_match = sum((df_trans["mo1"] == s1) & (df_trans["mo2"] == s2))
            num_all = sum(df_trans["mo1"] == s1)
            if num_all > 0:
                roll_rates.loc[s2,s1] = num_match / float(num_all)
    return roll_rates  

def getEmissionProb(observations,labels,states):
    GMM = {}
    for state in states:
        values = observations[labels==state]
        stateValues = []
        for column in values.columns:
            listvalue = values[column].tolist()
            l = [x for x in listvalue if ~np.isnan(x)]
            stateValues.extend(l)
        if(debug==1):
            plt.figure()
            plt.title(state)
            plt.hist(stateValues,10,normed=1)
        d1 = NormalDistribution(np.mean(stateValues), np.std(stateValues))
        d2 = NormalDistribution(0, 1)
        d3 = NormalDistribution(1, 1)
        clf = GeneralMixtureModel([d1, d2])
        clf.fit(np.array([[x] for x in stateValues]))
        if(debug==1):
            x = np.linspace(-8,8,500)
            y = clf.probability(x)
            plt.plot(x,y,'r')
            plt.show()
        GMM[state] = clf
        del clf
    return list(GMM.values())
def seq2num(seq,states):
    dic = {}
    seq_list = list(np.array(seq)[0])
    for s in states:
        dic[s] = states.index(s)
    seq_new = [[dic.get(n, n) for n in seq_list]]
    return seq_new[0]
def num2seq(seq,states):
    dic = {}
    seq_list = list(seq)
    for s in states:
        dic[states.index(s)] = s
    seq_new = [[dic.get(n, n) for n in seq_list]]
    return seq_new[0]
def HMM_predict(model,seq,algorithm='viterby'):
        result = model.predict(np.array(seq)[0], algorithm=algorithm)  
        if(algorithm=='viterby' and len(result)>len(np.array(seq)[0])):
            del result[0]
        return result
def getSeqConfusionMatrix(actual,predicted,states):
    y_actu = pd.Series(num2seq(actual,states),name='Actual')
    y_pred = pd.Series(num2seq(predicted,states), name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'])
    df_conf_norm = df_confusion.div(df_confusion.sum(axis=1), axis=0) 
    return df_conf_norm
def getAllConfusionMatrix(ECG,ECGLabel,states,model,algorithm='viterby'):
    confs=[]
    for i in range(0,len(ECG)):
        statePrediction = HMM_predict(model,ECG.loc[[i]],algorithm=algorithm)
#        print('--------------------------------------------------------')
#        print(i)
#        print(num2seq(statePrediction,states))
#        print(len(num2seq(statePrediction,states)))
#        print('--------------------------------------------------------')
#        print(list(np.array(ECGLabel.loc[[i]])))
#        print(len(list(np.array(ECGLabel.loc[[i]]))))
        stateActual=seq2num(ECGLabel.loc[[i]],states)
        confs.append(getSeqConfusionMatrix(stateActual,statePrediction,states))
#        print(i)
#        print(getSeqConfusionMatrix(stateActual,statePrediction,states))
    conf = sum(confs) / len(ECG)
    return conf
def getGrad(df):
    df2 = pd.DataFrame(columns=df.columns)
    for i in range(0,len(df)):
#        import pdb; pdb.set_trace()
        signal = np.array(df.loc[[i]])[0]
        signalGrad = np.gradient(signal)                
        df2.loc[i] = signalGrad
    return df2

def getECGWavelet(df,wavelet='coif1',level=1):
    df2 = pd.DataFrame(columns=df.columns)
    for i in range(0,len(df)):
        cA, cW = pywt.swt(np.array(df.loc[[i]])[0], wavelet,level=level)[0]
        df2.loc[i] = cA
    return df2
ECGLabel = seperateBaselines(ECGLabel)
#ECG = getGrad(ECG)

#ECG = getPCA(ECG)
#ECG = getECGWavelet(ECG,wavelet='coif1',level=2)
#ECG2 = ECG.subtract(np.mean(ECG,axis=1),axis=0)
#ECG = ECG2.divide(np.std(ECG,axis=1),axis=0)

kf = KFold(n_splits = 10, shuffle = False)
confs = []
for train_index, test_index in kf.split(ECGLabel):
    trainLabel = ECGLabel.loc[train_index]
    trainY = ECG.loc[train_index]
    testLabel = ECGLabel.loc[test_index]
    testY = ECG.loc[test_index]
    trainLabel = trainLabel.reset_index(drop=True)
    testLabel = testLabel.reset_index(drop=True)
    trainY = trainY.reset_index(drop=True)
    testY = testY.reset_index(drop=True)
#ECGLabel.to_csv('test.csv')
    initialProb = getInitialProb(trainLabel,states)
#    initialProb = np.array([0.2,0.2,0.2,0.2,0.2])
    transitionProb = getTransitionProb(trainLabel,states).values
    emisionProb = getEmissionProb(trainY,trainLabel,states)
    model = HiddenMarkovModel.from_matrix(np.transpose(transitionProb), emisionProb, initialProb,verbose=True)
    #conf = getSeqConfusionMatrix(stateActual,statePrediction,states)
    conf = getAllConfusionMatrix(testY,testLabel,states,model,algorithm='viterby')
    confs.append(conf)
#    print(conf)
#    break
print(sum(confs)/10)