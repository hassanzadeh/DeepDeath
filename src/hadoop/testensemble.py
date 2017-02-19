#!/usr/bin/env python

import sys
import gc
import os
import numpy as np
import pickle
from sklearn.metrics import accuracy_score

from optparse import OptionParser

from sklearn.metrics import roc_curve, auc
from sklearn.datasets import load_svmlight_file

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from utils import parse_svm_light_data


def load_model(path):
    with open(path, 'rb') as f:
        lines = [line.strip() for line in f]
        content = '\n'.join(lines)

        classifier = pickle.loads(content)
        return classifier


def predict_prob(classifiers, X):
    """
    Given a list of trained classifiers,
    predict the probability of positive label.
    (Return the average obtained from all the classifiers)
    """
    preds= []
    for classifier in classifiers:
        preds.append(classifier.predict_prob(X))
    return sum(preds)/len(preds)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-m", "--model-path", action="store", dest="path",
                      default="../data/models", help="path where trained classifiers are saved")
    parser.add_option("-t", "--test-file", action="store", dest="testfile",
                      default="../data/NCHS_uni+bigram_10k_test.txt", help="path to test file")
    parser.add_option("-n", "--n_features", action="store", dest="n_features",
                      default=5000, help="number of features")
    
    options, args = parser.parse_args(sys.argv)

    files = [options.path + "/" +
             filename for filename in os.listdir(options.path) if filename.startswith('part')]
    X_y  =load_svmlight_file (options.testfile,options.n_features)
    X=X_y[0]
    y=X_y[1]
    print ('Num classifiers: '+str(len(files))) 
    for file_no in range(len(files)):
        file=files[file_no]
        print ('Classifier: '+file)
        classifier=[]
        gc.collect()
        classifier = load_model (file)
        y_test_prob = []
        y_test = []

        if (file_no == 0):
            y_prob=np.array(classifier.predict_proba(X))
        else:
            y_prob =y_prob+np.array( classifier.predict_proba( X))

   
    y_prob=y_prob.tolist()
    y_max_prob=[]
   
    for i in range(len(y_prob)):
        y_max_prob.append(y_prob[i].index(max(y_prob[i]))+1)

    acc=sum(np.array(y)==np.array(y_max_prob))/float(len(y_max_prob))*100.0

    print ('Acc: '+str(acc))


