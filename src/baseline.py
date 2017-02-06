from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from random import shuffle


#TODO: replace with correct filepath
trainfile= '../data/NCHS_bigram_train_filtered.svmlight'
testfile='../data/NCHS_bigram_test_filtered.svmlight'

X_train, y_train = load_svmlight_file(trainfile)
X_test, y_test = load_svmlight_file(testfile)


skip = 3
#subsampling with skip rate 'skip'
X_train, y_train = X_train[::skip],y_train[::skip]
X_test, y_test = X_test[::skip],y_test[::skip]

print('begin knn\n')
#Note: you have to set n_jobs large enough to spread the tasks, otherwise it will crash
#clf= KNeighborsClassifier(n_jobs=10)
clf= LogisticRegression(n_jobs=1)
print('Training\n')
clf.fit(X_train, y_train)
print('Predicting\n')
y_pred = clf.predict(X_test)

print(accuracy_score(y_test,y_pred))
