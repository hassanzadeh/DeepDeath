#!/opt/conda/bin/python
import sys
import pickle
from optparse import OptionParser
from lrsgd import LogisticRegressionSGD
from utils import parse_svm_light_line
from sklearn.ensemble import RandomForestClassifier
import string
import random
from sklearn.datasets import load_svmlight_file


parser = OptionParser()
parser.add_option("-n", "--n_estimator", action="store", dest="n_estimator",
                  default=10, help="num estimators", type="int")
parser.add_option("-f", "--feature-num", action="store", dest="n_feature",
                  default=10000,help="number of features", type="int")
options, args = parser.parse_args(sys.argv)

clf=RandomForestClassifier(options.n_estimator)
filename=''.join(random.choice(string.ascii_uppercase+string.digits) for _ in range(4))
f=open (filename,'w')
for line in sys.stdin:
    key, value = line.split("\t", 1)
    f.write(value)

    value = value.strip()
    X, y = parse_svm_light_line(value)

f.close()

X_y = load_svmlight_file(filename)
clf.fit(X_y[0],X_y[1])
pickle.dump(clf, sys.stdout)
