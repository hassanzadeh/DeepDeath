#!/usr/bin/env python

import sys
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
                      default="../data/NCHS_uni+bigram_10k_test_v3.txt", help="path to test file")
    
    options, args = parser.parse_args(sys.argv)

    files = [options.path + "/" +
             filename for filename in os.listdir(options.path) if filename.startswith('part')]
    classifiers = map(load_model, files)
    print ('Num classifiers: '+str(len(classifiers))) 
    y_test_prob = []
    y_test = []
    X_y  =load_svmlight_file (options.testfile)
    X=X_y[0]
    y=X_y[1]
    print len(set(y)) 
    y_prob=np.array(classifiers[0].predict_proba(X))

    for i in xrange(1,len(classifiers)):
    	y_prob =y_prob+np.array( classifiers[i].predict_proba( X))
    y_prob=y_prob.tolist()


    y_max_prob=[]
   
    for i in range(len(y_prob)):
        y_max_prob.append(y_prob[i].index(max(y_prob[i])))
    y_pred=classifiers[0].predict(X)

    zp=zip(y_pred,y_max_prob)
    indx2label=[0]*len(y_prob[0])
    for pair in zp:
        indx2label[pair[1]]=pair[0]

    count =0 
    for i in range(len(y)):
        if (y[i]==indx2label[y_max_prob[i]]):
            count +=1
    acc=count/float(len(y))


    print ('Acc: '+str(acc))


