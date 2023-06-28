#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
#pd.set_option('display.width', 100)
#pd.set_option('max_columns', 40)
#pd.set_option('precision', 1)
from standard_vis import *


import seaborn as sns; sns.set()
sns.set(color_codes=True)
sns.set(font_scale=1.2)

import os
from subprocess import call
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import ast
import multiprocessing as mp
#from matplotlib import cm
from copy import copy
from numpy import arange
from numpy import random

from scipy.spatial.distance import cdist,pdist
from scipy.stats import levene, pearsonr
from scipy import stats
from scipy.stats import uniform

from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import utils
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import VarianceThreshold

from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.utils.multiclass import unique_labels

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterSampler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline, make_union
from sklearn.decomposition import PCA
from sklearn.datasets import make_regression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, jaccard_score, hamming_loss, confusion_matrix
# CLASSIFIERS
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

#REGRESSORS
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor

import statistics
from scipy.spatial.distance import cdist, pdist
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from sklearn import (manifold, decomposition, ensemble,
                     discriminant_analysis, random_projection)

#plt.style.use('seaborn')

import warnings
warnings.filterwarnings('ignore')


# Split data into 80% training data and 20% testing data
def data_split(X_features, y_labels):
    X_scaled = preprocessing.MinMaxScaler().fit_transform(X_features)
    X_norm = Normalizer().fit_transform(X_scaled)
    if y_labels.ndim > 1:
        y_multilabel_scaled = preprocessing.MinMaxScaler().fit_transform(y_labels)
    elif y_labels.ndim == 1:
        y_multilabel_scaled = (y_labels-min(y_labels))/(max(y_labels)-min(y_labels))
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y_multilabel_scaled, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Given optimized parameters and dataset, return a trained model and print its accuracy
def train_model(X_train, X_test, y_train, y_test, max_depth=91, min_samples_leaf=1, min_samples_split=4, n_estimators=800, criterion='mse', max_features='sqrt', bootstrap=False):  
    model_RFR = RandomForestRegressor(random_state=42, max_leaf_nodes=None, max_features=max_features,n_jobs=-1,
                                     criterion=criterion,max_depth=max_depth,bootstrap=bootstrap,
                                     min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                     n_estimators=n_estimators)

    model_RFR.fit(X_train, y_train)
    
    

    
    cv_scores = -cross_val_score(model_RFR, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
#     mse_scores = cross_val_score(model_RFR, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
#     mae_scores = cross_val_score(model_RFR, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')

    accuracy = np.mean(cv_scores)
    uncertainty = np.std(cv_scores)*2
    
    
    
#     training_score = model_RFR.score(X_train, y_train, scoring='neg_mean_squared_error')
#     testing_score = model_RFR.score(X_test, y_test, scoring='neg_mean_squared_error')

        
    training_score = mean_squared_error(y_train, model_RFR.predict(X_train))
    testing_score = mean_squared_error(y_test, model_RFR.predict(X_test))
    rmse = mean_squared_error(y_test, model_RFR.predict(X_test), squared=False)
    mae = mean_absolute_error(y_test, model_RFR.predict(X_test))
#     print(mean_squared_error(y_test, model_RFR.predict(X_test)))

    print('Training score:', np.round(training_score, 6))
    #print('CV Scores:', np.round(cv_scores, 6))
    print('CV Accuracy:',np.round(accuracy, 6),'+/-',np.round(uncertainty, 6))
    print('Testing score:', np.round(testing_score, 6))
    print('RMSE:', np.round(rmse, 6))
    print('MAE:', np.round(mae, 6))
    return model_RFR



# load data from csv file
def load_data(filename='data.csv'):
    X_df = pd.read_csv(filename)
    X_df = X_df.fillna(0)
    return X_df


def preprocess_data(data_frame):
    data_frame = data_frame.drop(['ID', 'Shape', 'Class'], axis=1)
    data_frame = data_frame.sort_index(axis = 0) 
    data_frame = data_frame.loc[:, (data_frame != 0).any(axis=0)]
    data_frame = data_frame.drop([30,31,34,107,199,301])
    data_frame=data_frame.loc[:, data_frame.std() > .001]
    data_frame.describe()
    return data_frame

#Visualize the tree
def tree_visualization(model, feature_names, filename): 
    out_file=filename+'.dot'
    save_file=filename+'.png'
    export_graphviz(model, 
                    feature_names=feature_names,
                    filled=True,
                   out_file=out_file)
    call(['dot', '-Tpng', out_file, '-o', save_file])
    img = mpimg.imread(save_file)
    imgplot = plt.imshow(img)
    plt.show()

    
def get_para(X_train, y_train):
    # A parameter grid to narrow down the range of paremeters
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 100)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(0, 100, num = 100)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [int(x) for x in np.linspace(start = 0, stop = 20, num = 100)]
    # Minimum number of samples required at each leaf nodes
    min_samples_leaf = [int(x) for x in np.linspace(start = 0, stop = 20, num = 100)]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    criterion=['friedman_mse','mse','mae']

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'criterion': criterion}

    model = RandomForestRegressor(random_state=42, n_jobs=-1) #make sure to use the same random state when you train the final model

    reg = RandomizedSearchCV(model, random_grid, cv=5, n_iter=1000)

    reg.fit(X_train, y_train)
    print(reg.best_params_)

# Data preparation
def data_preparation():
    MXene = MXene.strip()
    MXene = MXene.replace(' ', '')
    elements = MXene.split('2', 1)
    M = elements[0]
    X = elements[1][0]
    T = elements[1][1:]
    print(M, X, T)


# Evaluation
def multi_output_score(test, pred):
    true = 0
    for i in range(len(test)):
        if test[i]==pred[i]:
            true = true + 1
    return true/len(test)

def evaluation(true, pred):
    index_abs = 0
    jaccard_score_list = []
    hamming_score_list = []
    f1_score_list = []
    precision_score_list = []
    recall_score_list = []
    pred = pred.tolist()
    for row in true:
        single_score = jaccard_score(row, pred[index_abs], average='micro')
        hammung_score = hamming_loss(row, pred[index_abs])
        recall = recall_score(row, pred[index_abs], average='micro')
        precision = precision_score(row, pred[index_abs], average='micro')
        f1 = f1_score(row, pred[index_abs], average='micro')
        jaccard_score_list.append(single_score)
        hamming_score_list.append(hammung_score)
        precision_score_list.append(precision)
        recall_score_list.append(recall)
        f1_score_list.append(f1)
        index_abs = index_abs + 1
    return jaccard_score_list, hamming_score_list, f1_score_list, precision_score_list, recall_score_list