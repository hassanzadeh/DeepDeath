from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_svmlight_file

#Stratified subsampling
def subsampling(X, y, rate = 0.05):
    skf = StratifiedKFold(n_splits=int(1/rate))
    for train, test in skf.split(range(len(y)), y):
        new_X= X[test]
        new_y= y[test]
        break
    return new_X, new_y

#TODO: replace with correct filepath
trainfile= 'FILEPATH/NCHS_bigram_train.txt'
testfile='FILEPATH/NCHS_bigram_test.txt'
X_train, y_train = load_svmlight_file(trainfile)
X_test, y_test = load_svmlight_file(testfile)

#subsampling 10% if you want subsampling, or just remove the following two lines
X_train, y_train = subsampling(X=X_train,y=y_train)
X_test, y_test = subsampling(X=X_test,y=y_test)

print('begin knn\n')
#Note: you have to set n_jobs large enough to spread the tasks, otherwise it will crash
knn = KNeighborsClassifier(n_jobs=10)
knn.fit(X_train, y_train)
print('predict\n')
y_pred = knn.predict(X_test)

print(accuracy_score(y_test,y_pred))