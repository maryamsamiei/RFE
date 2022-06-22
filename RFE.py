#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 23:48:18 2021

@author: maryamsamieinasab
"""

import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score,RepeatedStratifiedKFold, KFold, RepeatedKFold,train_test_split
from sklearn.feature_selection import RFECV, RFE, SelectFromModel
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB,  MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve, plot_roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
import time
import argparse
import random
import matplotlib.pyplot as plt



def parse_args():
    """
    Parses the Meta_Server arguments.
    """
    parser = argparse.ArgumentParser(description="meta_server arguments")
    parser.add_argument('--savepath', nargs='?', default='./',help='save path for output')
    parser.add_argument('--input_train', nargs='?',help='pEA matrix of training set with genes of ineterest/last column is the labels')
    parser.add_argument('--input_test', nargs='?',help='pEA matrix of test set with genes of ineterest/ last column is the labels')
    parser.add_argument('--genes', nargs='?',help='list of genes of interest')
    parser.add_argument('--model', default='all', choices=('all','LR', 'SVM', 'DT', 'RF', 'GB', 'NB', 'AB'),help='which classifier to be trained on')
    parser.add_argument('--SVM', default='linear', choices=('linear', 'gaussian', 'poly'),help='which kernel to be used in SVM classifer')
    parser.add_argument('--NB', default='gaussian', choices=('gaussian', 'multinomial'),help='which NB classifier to be used based on the prior distribution')
    parser.add_argument('--LR', default='elasticnet', choices=('elasticnet', 'l1'),help='which penalty to be used in LR classifer')
    # parser.add_argument('--maxf',type=int, default='200',help='maximum # of selected features')
    # parser.add_argument('--minf',type=int, default='100',help='minimum # of selected features')
    parser.add_argument('--balance',type=int, default='0',choices=('0', '1'),help='1  for balancing the validation split, O/W 0')
    return parser.parse_args()
def Recursive_feature_elimination(nfeatures, estimator, X_train,y_train):
    rfe = RFE(estimator=estimator,n_features_to_select=nfeatures)
    rfe.fit(X_train,y_train)
    X_train_rfe = rfe.transform(X_train)
    estimator.fit(X_train_rfe,y_train)
    rfe_ranking = pd.DataFrame(rfe.ranking_)
    return rfe_ranking

def Balance_classes(Y):
    cases_idx = np.where(Y == 1)[0].tolist()
    controls_idx = np.where(Y == 0)[0].tolist()
    min_idx = np.argmin((len(cases_idx), len(controls_idx)))
    smaller_group_idx, larger_group_idx = (cases_idx, controls_idx) if min_idx == 0 else (controls_idx, cases_idx)
    larger_sub_idx = random.sample(larger_group_idx, len(smaller_group_idx))
    new_class_idx = sorted(smaller_group_idx + larger_sub_idx)
    return new_class_idx 
def get_stacking():
	# define the base models
    level0 = list()
    level0.append(('LR', LogisticRegression(penalty='l1',solver='saga',max_iter=5000)))
    level0.append(('Elasticnet',  LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)))
    level0.append(('RF',RandomForestClassifier(n_estimators =1000, max_depth=5,random_state=0, oob_score=True, n_jobs=-1)))
    level0.append(('AB', AdaBoostClassifier(n_estimators=1000, random_state=0)))
    level0.append(('GB',GradientBoostingClassifier(n_estimators=1000, random_state=0)))
    level0.append(('SVM',SVC(kernel='linear') ))
    level1 = LogisticRegression()
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model
def get_models():
    models = dict()
    models['RF'] = RandomForestClassifier(n_estimators =1000,max_depth=5,random_state=18401, n_jobs=-1)
    models['AB']= AdaBoostClassifier(n_estimators=1000)
    models['svm'] = SVC(kernel='linear') 
    models['LR'] = LogisticRegression(penalty='l1',solver='saga',max_iter=5000)
    models['Elasticnet'] = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5) 
    models['GB'] = GradientBoostingClassifier(n_estimators=1000, random_state=0)
    models['stacking']=get_stacking()
    return models

def plot_acc_curve(accuracy_test, savepath, model,name, X_train, y_train, X_test, y_test, genes):
    name=str(name)
    ranking = pd.read_csv(savepath+name+'_ranking.csv', header=None) 
    
    for j in range(len(genes)):
        features = np.where(ranking.iloc[:,0] <j+2)
        top_features=[]
        for i in features[0]:
            top_features+=[genes[i]]

        X_train_top_features = X_train[top_features]
        X_test_top_features = X_test[top_features]
      
        model.fit(X_train_top_features,y_train)
        y_pred = model.predict(X_test_top_features)
        accuracy_test.loc[j,name] = accuracy_score(y_test, y_pred)
    
    idx = accuracy_test.loc[0:200,name].idxmax()
    plt.clf()
    plt.close('all')
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(accuracy_test[name], label=name)
    ax.axvline(x=idx, ymin=0, ymax=1, color='red', label=str(idx))
    plt.ylabel('Accuracy',fontsize=18)
    plt.xlabel('# of selected genes', fontsize=18)
    ax.legend(loc='upper left')
    fig.savefig(savepath+name+'_ACC.png')
    return idx
def classification(X_train, y_train, X_test, y_test):
    # models = get_models()
    # for name, model in models.items():
    #     name = str(name)
    #     model.fit(X_train,y_train)
    #     y_pred = model.predict(X_test)
    #     acc = accuracy_score(y_test, y_pred)
    #     fig= plot_roc_curve(model, X_test, y_test)
    #     AUC = fig.roc_auc
    #     text_file.write('RFE model: '+str(RFE_model)+'.....'+'Classifier: '+name+'\n'+'Accuracy= '+str(acc)+'\t'+'AUC= '+str(AUC)+'\n')
   
    # model = get_stacking()
    # name = 'stacking'
    model= RandomForestClassifier(n_estimators = 1000, max_depth=5, random_state=18401)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    fig= plot_roc_curve(model, X_test, y_test)
    AUC = fig.roc_auc
    return acc, AUC
    #text_file.write('RFE model: '+str(RFE_model)+'.....'+'Classifier: '+name+'\n'+'Accuracy= '+str(acc)+'\t'+'AUC= '+str(AUC)+'\n')
    
 
def main(args):
    start_time = time.time()
    print('analysis started')
    X_train = pd.read_csv(args.input_train,index_col=0)
    y_train = X_train['label']
    X_val = pd.read_csv(args.input_test,index_col=0)
    y_val = X_val['label']
    if args.balance==1:
        case_control_idx = Balance_classes(Y=y_val)
        X_val = X_val.iloc[case_control_idx]
        case_control_idx_train = Balance_classes(Y=y_train)
        X_train = X_train.iloc[case_control_idx_train]

    genes = pd.read_csv(args.genes,header=None)
    genes = genes.iloc[:,0].tolist()
    X_train = X_train[genes]
    X_val = X_val[genes]
    X_train1 = X_train
    y_train1 = y_train
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=1)
    print('Number of samples in training set ', X_train.shape[0],'\n',
          'Number of input genes are ',len(genes), '\n', 'RFE will be trained on ', args.model)
    nfeatures=1
    accuracy_test = pd.DataFrame(np.zeros((len(genes), 6)), columns=('RF','AB','SVM','GB','LR','Elasticnet'))
    if args.model=='all':
        models = get_models()
        for name, model in models.items():
            rfe_ranking = Recursive_feature_elimination(nfeatures,model, X_train, y_train)    
            rfe_ranking.to_csv(args.savepath+str(name)+'_ranking.csv',index=False, header=False)
            idx = plot_acc_curve(accuracy_test, args.savepath, model,name, X_train, y_train, X_test, y_test, genes)
            features = np.where(rfe_ranking.iloc[:,0] <idx+1)
            top_features=[]
            for i in features[0]:
                top_features+=[genes[i]]
            print ('RFE model ', str(name), 'is done.')
        X_train_top_features = X_train1[top_features]
        X_val_top_features = X_val[top_features]
        acc, AUC = classification(X_train_top_features, y_train1, X_val_top_features, y_val)
        text_file = open(args.savepath+'performance_summary.txt', 'w')
        text_file.write('RFE model: '+str(name)+'.....'+'Classifier: '+'RF'+'\n'+'Accuracy= '+str(acc)+'\t'+'AUC= '+str(AUC)+'\n')   
        text_file.close()     
    else:
        
        if args.model=='LR':
            if args.LR == 'l1':
                estimator=LogisticRegression(penalty='l1',solver='saga',max_iter=5000)
            elif args.LR == 'elasticnet':    
                estimator = LogisticRegressionCV(penalty='elasticnet', solver="saga",l1_ratios=np.arange(0, 1, 0.1).tolist(),cv=5)  # Stochastic Average Gradient descent solver.
        if args.model == 'DT':
            estimator=DecisionTreeClassifier()
        if args.model== 'RF':
            estimator = RandomForestClassifier(n_estimators =1000, random_state=0, oob_score=True, n_jobs=-1)
        if args.model== 'SVM':
            if args.SVM =='linear':
                estimator = SVC(kernel="linear")
            elif args.SVM == 'poly':
                estimator = SVC(kernel="poly",degree=8)
            elif args.SVM== 'gaussian':
                estimator = SVC(kernel='rbf')   
        if args.model== 'GB':
            estimator = GradientBoostingClassifier(n_estimators=100, random_state=0)
        if args.model=='NB':
            if args.NB == 'gaussian':
                estimator = GaussianNB()
            elif args.NB == 'multinomial':
                estimator = MultinomialNB()
        if args.model=='AB':
            estimator= AdaBoostClassifier(n_estimators=1000,random_state=0)

        rfe_ranking = Recursive_feature_elimination(nfeatures,estimator, X_train, genes)
        rfe_ranking.to_csv(args.savepath+args.model+'_ranking.csv',index=False, header=False)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    args = parse_args()
    main(args)