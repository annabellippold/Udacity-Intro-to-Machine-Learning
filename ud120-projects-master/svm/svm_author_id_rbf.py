#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

""" import """
from sklearn.svm import SVC

""" create classifier (linear kernel) """
clf = SVC(kernel="rbf", C= 10000.0)

### make the dataset smaller to speed up
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

""" fit / train data """
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

""" make predictions """
t0 = time()
pred = clf.predict(features_test)
print "training time:", round(time()-t0, 3), "s"
print(pred)
print("10: ", pred[10])
print("26: ", pred[26])
print("50: ", pred[50])
"""Results
('10: ', 1)
('26: ', 0)
('50: ', 1)
"""
answer = 0
for c in pred:
	if c == 1:
		answer +=1

print answer
# 1018 of Chris reduced dataset
# 877 of Chris full dataset

""" calculate accuracy """
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)
print accuracy
print clf.score(features_test, labels_test)
#########################################################

""" Results with smaller dataset:
        no. of Chris training emails: 7936
        no. of Sara training emails: 7884
        training time: 0.098 s
        training time: 1.04 s
        0.616040955631
        0.616040955631
        
        accurcy for :
            10:     0.616040955631
            100:    0.616040955631
            1000:   0.821387940842
            10000:  0.892491467577
    
    Results with a full dataset and C = 10000:
        training time: 100.055 s
        training time: 10.159 s
        0.990898748578
"""
