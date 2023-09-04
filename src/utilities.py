#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
#pd.set_option('display.width', 100)
#pd.set_option('max_columns', 40)
#pd.set_option('precision', 1)
from src.standard_vis import *


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
from torch.utils.data import Dataset, TensorDataset
import torch
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

def preprocess(filename='../data/data_withLattice.csv', elemental_descriptor=True):
    X_df = load_data(filename)
    X_features = X_df.iloc[:, 1:-5]
    feature_names = X_features.columns.values

    y_multilabel = X_df.iloc[:, -5:]
    label_names = y_multilabel.columns.values

    X = X_features.to_numpy()
    y = y_multilabel.to_numpy()
    scaler = preprocessing.MinMaxScaler()
    y = scaler.fit_transform(y)

    data_index = []
    for index, row in X_features.iterrows():
        indices = [i for i, x in enumerate(row) if x == 1]
        indices[1] = indices[1] - 9
        indices[2] = indices[2] - 11
        indices[3] = indices[3] - 16
        data_index.append(indices)
    X_features_classes = np.array(data_index)

    if elemental_descriptor:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_features_classes, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.Tensor(y_train)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    X_test_tensor = torch.Tensor(X_test)
    y_test_tensor = torch.Tensor(y_test)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    X_val_tensor = torch.Tensor(X_val)
    y_val_tensor = torch.Tensor(y_val)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
    return train_dataloader, test_dataloader, val_dataloader

def m_training_process_vis(points_all_max, points_all_min, vlist, save=False, suffix=''):
    fig = plt.figure(figsize=[6,6])
    ax = fig.add_subplot(111)
    colors = cm.rainbow(np.linspace(0, 1, len(vlist)))
    points_all = []
    for idx, i in enumerate(range(len(points_all_min))[::-1]):
        points_single = points_all_max[idx]+points_all_min[idx]
        points_sorted = []
        for p in points_single:
            points_sorted += p
        points_all.append(points_sorted)

    for idx, i in enumerate(points_all):
        i=np.array(i)
        xx, xy = zip(*sorted(zip(i[:,0],i[:,1])))
        ax.plot(xx, xy, marker='o', color = colors[idx], label=vlist[idx])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color ="black", linestyle='--', lw = 1, alpha=0.5)
    ax.set_facecolor("white")
    ax.legend(bbox_to_anchor=(1, 1.02), shadow=False, facecolor='white', fontsize=12)
    ax.set_xlabel('Individual mask', fontsize=18)
    ax.set_ylabel('Loss difference', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    for location in ['left', 'right', 'top', 'bottom']:
        ax.spines[location].set_linewidth(1)
        ax.spines[location].set_color('black')
    if save:
        plt.savefig('../results/figs/exploring_process_{}.png'.format(suffix), bbox_inches='tight')
    plt.show()