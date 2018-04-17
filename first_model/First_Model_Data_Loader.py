# Load pickled data
import pickle
import csv
import numpy as np

training_file = './data.p'
testing_file = './data.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
#print(train)
X_train, y_train = train['features'], train['labels']
X_train = np.array(X_train)
y_train_ = np.array(y_train)


for i in range(0,len(y_train)):
    y_train[i]=int(y_train_[i])
y_train=np.array(y_train)
X_test, y_test = test['features'], test['labels']
X_test = np.array(X_test)
y_test = np.array(y_test)

# read csv to get sign names
sign_names = []
with open('First_Model_signnames.csv') as signname_file:
    signname_reader = csv.DictReader(signname_file)
    sign_names = [row['SignName'] for row in signname_reader]