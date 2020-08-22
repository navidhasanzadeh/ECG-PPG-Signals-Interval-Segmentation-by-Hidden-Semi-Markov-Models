# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pywt
from pomegranate import GeneralMixtureModel
import pandas as pd
from pomegranate import *
import copy
import re
from hsmmlearn.hsmm import HSMMModel
from sklearn.decomposition import PCA

import networkx as nx
from sklearn.model_selection import KFold
import hsmmlearn.hsmm
from hsmmlearn.emissions import GaussianEmissions
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from hsmmlearn.emissions import AbstractEmissions
from scipy.stats import norm
from scipy.stats import multivariate_normal

debug = 0
IDs = pd.read_csv(r"out\id-1.csv",header=None)
ECG = pd.read_csv(r"out\PPG-1.csv",header=None)
ECGLabel = pd.read_csv(r"out\PPGLabel-1.csv",header=None)


#ECG = ECG.drop([248,249],axis=1)
#ECGLabel = ECGLabel.drop([248,249],axis=1)
states = ['A','B','X','C','D']
#states = ['B','P','R','T']

class multiGaussianEmissionsGMM(AbstractEmissions):
#    import pdb; pdb.set_trace()

    dtype = np.float64
    def __init__(self, observations,labels,states):
        self.means = 0
        self.scales = 0
        GMM = {}
        for state in states:
            values = [obs[labels==state] for obs in observations]
            stateValues = {i:[] for i in range(0,len(values))}
            i=-1
            for obsValue in values:
                i = i + 1
                for column in obsValue.columns:
                    listvalue = obsValue[column].tolist()
                    l = [x for x in listvalue if ~np.isnan(x)]
                    stateValues[i].extend(l)        
            obsDists = []
            i=-1
            for obsValue in values:
                i = i + 1
                d1 = NormalDistribution(np.mean(stateValues[i]), np.std(stateValues[i]))
                d2 = NormalDistribution(1, 1)
                d3 = NormalDistribution(-1, 1)
                clf=d1
#                d1.get_params
#                clf = GeneralMixtureModel([d1,d2])
                clf.fit(np.array([[x] for x in stateValues[i]]))
                obsDists.append(clf)                
#                plt.figure()
#                plt.title(state)
#                plt.hist(stateValues[i],10,normed=1)
#                x = np.linspace(-8,8,500)
#                y = clf.probability(x)
#                plt.plot(x,y,'r')
#                plt.show()
#            dist = IndependentComponentsDistribution(obsDists)          
#            dist.fit(np.transpose([x for x in stateValues.values()]))            
            GMM[state] = obsDists
        self.GMM = list(GMM.values())
        
    def likelihood(self, obs):
#        obs = np.squeeze(obs)        
        result = []
        for r in self.GMM:
            prob = 1
            i=0
            for dist in r:
#                import pdb; pdb.set_trace()
                x = np.linspace(-10,10,10000);
                y = np.sum(0.0001 * np.array(dist.probability(x)))
                prob = prob * dist.probability(np.transpose(obs)[i])
                i = i + 1
            result.append(prob)
        return np.array(result)

    def sample_for_state(self, state, size=None):        
        return self.GMM[state].sample()

    def copy(self):
        return multiGaussianEmissionsGMM(self.means.copy(), self.scales.copy())


def getECGWavelet(df,wavelet='coif1',level=1):
    df2 = pd.DataFrame(columns=df.columns)
    for i in range(0,len(df)):
        cA, cW = pywt.swt(np.array(df.loc[[i]])[0], wavelet,level=level)[0]
        df2.loc[i] = cA
    return df2
def getGrad(df):
    df2 = pd.DataFrame(columns=df.columns)
    for i in range(0,len(df)):
#        import pdb; pdb.set_trace()
        signal = np.array(df.loc[[i]])[0]
        signalGrad = np.gradient(signal)                
        df2.loc[i] = signalGrad
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
def getDurationDistEstimator(df,states):
    count = {s:[] for s in states}    
    for i in range(0,len(df)):
        for state in states:
            sep = '[' + state + ']+'
            dfRow = ''.join(ECGLabel.loc[i].tolist())
            x = re.findall(sep,dfRow)
            x = [len(y) for y in x]
            if(dfRow[0]==state):
                del x[0]
            if(dfRow[-1]==state):
                del x[-1]
            count[state].extend(x)
    durationDists = {}
    for state in states:
        gd = GammaDistribution.from_samples(np.array(count[state]))
        x = np.linspace(0,100,101)
        y = gd.probability(x)
        y = y / (np.sum(y))
#        print(np.sum(y))
        if(debug==1):
            plt.figure()
            plt.title(state)
            plt.hist(np.array(count[state]),10,normed=1)                     
            plt.plot(x,y,'r')
            plt.show()
        durationDists[state] = y
    durationDists = np.array([list(y) for y in durationDists.values()])
    return durationDists
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
                x[0] = x[0].replace('B','Q')
                y = ''.join(x)    
                result = result + y
            else:
                result = result + x[0]
        result = list(result)
        df2.loc[i] = result
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
    return np.transpose(roll_rates)

def getEmissionProb(observations,labels,states):
    means = []
    stds = []
    for state in states:
        values = observations[labels==state]
        stateValues = []
        for column in values.columns:
            listvalue = values[column].tolist()
            l = [x for x in listvalue if ~np.isnan(x)]
            stateValues.extend(l)        
        means.append(np.mean(stateValues))
        stds.append(np.std(stateValues))        
    dists = GaussianEmissions(np.array(means), np.array(stds))
    return dists
def getEmissionProb(observations,labels,states):
    means = []
    stds = []
    for state in states:
        values = [obs[labels==state] for obs in observations]
        stateValues = {i:[] for i in range(0,len(values))}
        i=-1
        for obsValue in values:
            i = i + 1
            for column in obsValue.columns:
                listvalue = obsValue[column].tolist()
                l = [x for x in listvalue if ~np.isnan(x)]
                stateValues[i].extend(l)        
        obsDists = []
        i=-1
        mean = []
        std = []
        for obsValue in values:
            i = i + 1
            mean.append(np.mean(stateValues[i]))
            std.append(np.std(stateValues[i]))            
        means.append(mean)
        stds.append(std)
    dist = multiGaussianEmissions(means,stds)
    return dist
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
#        import pdb; pdb.set_trace()
#        print(seq)
        result = model.decode(np.transpose([np.array(x)[0] for x in seq]))
#        if(algorithm=='viterby' and len(result)>len(np.array(seq)[0])):
#            del result[0]
        return result
def getSeqConfusionMatrix(actual,predicted,states):
    y_actu = pd.Series(num2seq(actual,states),name='Actual')
    y_pred = pd.Series(num2seq(predicted,states), name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'])
    df_conf_norm = df_confusion.div(df_confusion.sum(axis=1), axis=0) 
    return df_conf_norm
def getAllConfusionMatrix(ECG,ECGLabel,states,model,algorithm='viterby'):
    confs=[]
    for i in range(0,len(ECG[0])):
        statePrediction = HMM_predict(model,[x.loc[[i]] for x in ECG],algorithm=algorithm)
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
    conf = sum(confs) / len(ECG[0])
    return conf
ECGLabel = seperateBaselines(ECGLabel)

ECG2 = getGrad(ECG)
ECG2 = ECG2.subtract(np.mean(ECG2,axis=1),axis=0)
ECG2 = ECG2.divide(np.std(ECG2,axis=1),axis=0)
ECG3 = getECGWavelet(ECG,wavelet='sym2',level=3)
ECG3 = ECG3.subtract(np.mean(ECG3,axis=1),axis=0)
ECG3 = ECG3.divide(np.std(ECG3,axis=1),axis=0)
ECG4 = getECGWavelet(ECG,wavelet='coif1',level=3)
ECG4 = ECG4.subtract(np.mean(ECG4,axis=1),axis=0)
ECG4 = ECG4.divide(np.std(ECG4,axis=1),axis=0)
#ECGmin = np.min(ECG,axis=1)
#ECGmax = np.max(ECG,axis=1)
#ECG = ECG.subtract(ECGmin,axis=0)
#ECG = ECG.divide(ECGmax-ECGmin,axis=0)
#ECG2 = getGrad(ECG)
#ECG2 = ECG2.subtract(np.mean(ECG2,axis=1),axis=0)
#ECG2 = ECG2.divide(np.std(ECG2,axis=1),axis=0)
#ECG3 = getGrad(ECG2)
#ECG3 = ECG3.subtract(np.mean(ECG3,axis=1),axis=0)
#ECG3 = ECG3.divide(np.std(ECG3,axis=1),axis=0)
#ECG2 = getECGWavelet(ECG,level=1)
#ECG = getPCA(ECG)
kf = KFold(n_splits = 10, shuffle = False)
confs = []
for train_index, test_index in kf.split(ECGLabel):
    trainLabel = ECGLabel.loc[train_index]
    trainY = ECG.loc[train_index]
    trainY2 = ECG2.loc[train_index]
    trainY3 = ECG3.loc[train_index]
    trainY4 = ECG4.loc[train_index]
    testLabel = ECGLabel.loc[test_index]
    testY = ECG.loc[test_index]
    testY2 = ECG2.loc[test_index]
    testY3 = ECG3.loc[test_index]
    testY4 = ECG4.loc[test_index]
    trainLabel = trainLabel.reset_index(drop=True)
    testLabel = testLabel.reset_index(drop=True)
    trainY = trainY.reset_index(drop=True)
    trainY2 = trainY2.reset_index(drop=True)
    trainY3 = trainY3.reset_index(drop=True)
    trainY4 = trainY4.reset_index(drop=True)
    testY = testY.reset_index(drop=True)
    testY2 = testY2.reset_index(drop=True)
    testY3 = testY3.reset_index(drop=True)
    testY4 = testY4.reset_index(drop=True)
#ECGLabel.to_csv('test.csv')
    initialProb = getInitialProb(trainLabel,states)
#    initialProb = np.array([0.2,0.2,0.2,0.2,0.2])
    transitionProb = getTransitionProb(trainLabel,states).values
    emisionProb = multiGaussianEmissionsGMM([trainY2],trainLabel,states)
    durations = getDurationDistEstimator(trainLabel,states)
    model = HSMMModel(emisionProb,durations,transitionProb,initialProb)
#    model = HiddenMarkovModel.from_matrix(transitionProb, emisionProb, initialProb,verbose=True)
    #conf = getSeqConfusionMatrix(stateActual,statePrediction,states)
    conf = getAllConfusionMatrix([testY2],testLabel,states,model,algorithm='viterby')
    confs.append(conf)
#    break
#    print(conf)
print(sum(confs)/10)