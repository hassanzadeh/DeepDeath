from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
from sklearn.datasets import dump_svmlight_file,load_svmlight_file

filepath = 'FILEPATH' #replace with correct file path

#Preprocess
fname = filepath+'/NCHS_tabular.txt'
with open(fname) as f:
    content = f.readlines()


patientid, label, data = [],[],[]
for i in content:
    elements = i.split()
    label.append(elements[1])
    patientid.append(elements[0])
    chain = []
    for j in range(2,len(elements)):
        chain.append(elements[j][3:])
    data.append(' '.join(chain))


#Load labels
fname = filepath+'/NCHS_train_or_test.txt'
with open(fname) as f:
    content = f.readlines()

train, test = {},{}
for i in content:
    elements=i.split()
    if elements[1]=='train':
        train[elements[0]]=1
    elif elements[1]=='test':
        test[elements[0]]=1
train_index, test_index=[],[]
for i in range(len(patientid)):
    if patientid[i] in test.keys():
        test_index.append(i)
    else:
        train_index.append(i)

#Generating unigrams
vectorizer = CountVectorizer(ngram_range=(1,1))

X = vectorizer.fit_transform(data).toarray()
print('Unigram generated\n')

X_train = X[train_index]
X_test = X[test_index]
y_train = [label[i] for i in train_index]
y_test = [label[i] for i in test_index]

dump_svmlight_file(X_train, np.array(y_train, dtype='int'), f=filepath+'NCHS_unigram_all_train.txt')
dump_svmlight_file(X_test, np.array(y_test, dtype='int'), f=filepath+'/NCHS_unigram_all_test.txt')


#Generating top 5k bigrams
vectorizer = CountVectorizer(ngram_range=(2,2),max_features=5000)

X = vectorizer.fit_transform(data).toarray()
print('Bnigram generated\n')

X_train = X[train_index]
X_test = X[test_index]
y_train = [label[i] for i in train_index]
y_test = [label[i] for i in test_index]

dump_svmlight_file(X_train, np.array(y_train, dtype='int'), f=filepath+'/NCHS_bigram_5k_train.txt')
dump_svmlight_file(X_test, np.array(y_test, dtype='int'), f=filepath+'/NCHS_bigram_5k_test.txt')


#Generating top 10k uni+bigrams

vectorizer = CountVectorizer(ngram_range=(1,2),max_features=10000)

X = vectorizer.fit_transform(data).toarray()
print('Uni+bigram generated\n')

dump_svmlight_file(X, np.array(label, dtype='int'), f=filepath+'/NCHS_uni+bigram_10k.txt')  #Prevent memory issues
X, label = load_svmlight_file(filepath+'/NCHS_uni+bigram_10k.txt')

X_train = X[train_index]
X_test = X[test_index]
y_train = [label[i] for i in train_index]
y_test = [label[i] for i in test_index]

dump_svmlight_file(X_train, np.array(y_train, dtype='int'), f=filepath+'/NCHS_uni+bigram_10k_train.txt')
dump_svmlight_file(X_test, np.array(y_test, dtype='int'), f=filepath+'/NCHS_uni+bigram_10k_test.txt')
