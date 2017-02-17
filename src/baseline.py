from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from random import shuffle
from sklearn.ensemble import RandomForestClassifier as RF
import sys


#TODO: replace with correct filepath
#trainfile= '../data/NCHS_bigram_train_filtered.svmlight'
#testfile='../data/NCHS_bigram_test_filtered.svmlight'
#trainfile= '../data/NCHS_bigram_train.txt'
#testfile='../data/NCHS_bigram_test.txt'
#trainfile= '../data/NCHS_unigram_train.txt'
#testfile='../data/NCHS_unigram_test.txt'
#trainfile= '../data/NCHS_unigram_all_train.txt'
#testfile='../data/NCHS_unigram_all_test.txt'



def train_test(train_test_file,train_size,skip=3):

    X, y= load_svmlight_file(train_test_file)
    X_train,y_train = X[:train_size],y[:train_size]
    X_test, y_test = X[train_size:],y[train_size:]
    #subsampling with skip rate 'skip'
    X_train, y_train = X_train[::skip],y_train[::skip]
    X_test, y_test = X_test[::skip],y_test[::skip]

    print('begin knn\n')
    #Note: you have to set n_jobs large enough to spread the tasks, otherwise it will crash
    #clf= KNeighborsClassifier(n_jobs=1)
    #clf= LogisticRegression(n_jobs=-1)
    clf= RF(n_estimators=5,n_jobs=1)
    print('Training\n')
    clf.fit(X_train, y_train)
    print('Predicting\n')
    y_pred = clf.predict(X_test)

    print(accuracy_score(y_test,y_pred))


if __name__ == '__main__':
    train,test,skip,train_size='','',3,0
    if (len(sys.argv)<4):
        skip=3
    else:
        skip=int(sys.argv[3])
    if (len(sys.argv)>2) :
        train_test(sys.argv[1],int(sys.argv[2]),skip)
    else:
        print ('Wrong arguments')
    
    	


