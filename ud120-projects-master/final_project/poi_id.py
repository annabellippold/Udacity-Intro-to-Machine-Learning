#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# --- Adding features to find POIs 
#    Selecting features by k-best, list with scores in select_features.py
features_list = ['poi','exercised_stock_options', 'total_stock_value', 'bonus']#,
                 #'salary']#, 'deferred_income', 'long_term_incentive',
                 #'restricted_stock']#, 'total_payments', 'shared_receipt_with_poi',
                 #'loan_advances', 'expenses', 'from_poi_to_this_person', 'from_this_person_to_poi']
         

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Information about data

# --- Length of Dataset
print "Count Dataset: ", len(data_dict) 
print "Count features: ", len(data_dict[data_dict.keys()[0]]) 
# The dataset has 145 rows and 23 columns (origianl 21 columns + 2 new added
# features).

# --- How many POIs are in the dataset? 
poi = 0
for key in data_dict:
    if data_dict[key]['poi'] == 1:
        poi =poi + 1
print "Count POIs: ", poi        
# There are 18 POIs in the dataset.                                 
          
# --- Test if keys are relevant
#print data_dict.keys()
# print a list of all names (keys) which should be employees of Enron
# The key 'THE TRAVEL AGENCY IN THE PARK' is not an employee

### Task 2: Remove outliers
# --- The Total Value from Salary should be removed.
data_dict.pop("TOTAL", 0)

# --- The key 'THE TRAVEL AGENCY IN THE PARK', which isn't an employee
#     should be deleted, it can influence the analysis on the features
#     total_payments and other.
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

# --- Remove Employee without data
#     By looking at the enronInsiderPay.pdf the Employee Eugen Lockhard has no
#     data and can be droped.
#print data_dict['LOCKHART EUGENE E']
"""{'to_messages': 'NaN', 'deferral_payments': 'NaN', 'expenses': 'NaN', 
    'poi': False, 'bonus_rate': 'NaN', 'deferred_income': 'NaN', 
    'email_address': 'NaN', 'long_term_incentive': 'NaN', 
    'restricted_stock_deferred': 'NaN', 'shared_receipt_with_poi': 'NaN', 
    'loan_advances': 'NaN', 'from_messages': 'NaN', 'other': 'NaN', 
    'poi_email_rate': 'NaN', 'director_fees': 'NaN', 'bonus': 'NaN', 
    'total_stock_value': 'NaN', 'from_poi_to_this_person': 'NaN', 
    'from_this_person_to_poi': 'NaN', 'restricted_stock': 'NaN', 
    'salary': 'NaN', 'total_payments': 'NaN', 'exercised_stock_options': 'NaN'}
"""
data_dict.pop("LOCKHART EUGENE E", 0)

# --- Remove NaN form salary; can be done for a lot of other features!
outliers = []
for key in data_dict:
    nan_salary = data_dict[key]['salary']
    if nan_salary == 'NaN':
        continue
    outliers.append((key, int(nan_salary)))

outliers_final = (sorted(outliers,key=lambda x:x[1],reverse=True)[0:10])
# Print people with top Salary
#print len(outliers_final)
# 94
"""
    [('SKILLING JEFFREY K', 1111258), ('LAY KENNETH L', 1072321), 
    ('FREVERT MARK A', 1060932), ('PICKERING MARK R', 655037), 
    ('WHALLEY LAWRENCE G', 510364), ('DERRICK JR. JAMES V', 492375), 
    ('FASTOW ANDREW S', 440698), ('SHERRIFF JOHN R', 428780), 
    ('RICE KENNETH D', 420636), ('CAUSEY RICHARD A', 415189)]
"""


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
sys.path.append("./final_project/")
from creat_new_features import poi_email_rate, bonus_rate

# --- Add new created features, creat_new_features.py
poi_email_rate(data_dict, features_list)
bonus_rate(data_dict, features_list)

### Store to my_dataset for easy export.
my_dataset = data_dict

data = featureFormat(my_dataset, features_list)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# --- feature scaling
#def scale_features(features):
#    """
#    Featrue sclaing with MinMax algorithm
#    """
#
#    from sklearn import preprocessing
#    scaler = preprocessing.MinMaxScaler()
#    features = scaler.fit_transform(features)
#
#    return features

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

# --- Splitting data into a Test and Training data set:
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

## --- Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)
print "Accuracy Naive Bayes: ", accuracy
#""" Accuracy with k-best features:
#Accuracy Naive Bayes:   0.921052631579
#"""

# --- SVM
#from sklearn.svm import SVC
##clf = SVC(kernel="rbf", C = 1000.)
#clf = SVC(C=1, cache_size=200, class_weight='balanced', coef0=0.0,
#  decision_function_shape=None, degree=3, gamma=0.1, kernel='rbf',
#  max_iter=-1, probability=False, random_state=None, shrinking=True,
#  tol=0.001, verbose=False)

#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)
#
#""" calculate accuracy """
#from sklearn.metrics import accuracy_score
#accuracy_svm = accuracy_score(pred, labels_test)
#print "Accuracy SVM: ", accuracy_svm
#""" Accuracy with k-best features:
#Accuracy SVM:  0.883720930233
#"""

# --- Decision Tree
#from sklearn import tree
##clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=1,
##            min_impurity_split=1e-07, min_samples_leaf=1,
##            min_samples_split=2, min_weight_fraction_leaf=0.0,
##            presort=False, random_state=42, splitter='best')
#clf = tree.DecisionTreeClassifier()
#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)
#
#from sklearn.metrics import accuracy_score
#accuracy_tree = accuracy_score(pred, labels_test)
#print "Accuracy Decision Tree: ", accuracy_tree
#""" Accuracy with k-best features:
#Accuracy Decision Tree: 0.813953488372
#"""

## --- adaboost
#from sklearn.ensemble import AdaBoostClassifier
#
##clf = AdaBoostClassifier()
#clf = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
#          learning_rate=1.0, n_estimators=1, random_state=42)
#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)
#
#from sklearn.metrics import accuracy_score
#accuracy_adaboost = accuracy_score(pred, labels_test)
#
#print "Accuracy AdaBoost: ", accuracy_adaboost
#""" Accuracy with k-best features:
#Accuracy AdaBoost:  0.837209302326
#"""

# --- Precision and Recall
import sklearn.metrics as m
precision = m.precision_score(pred, labels_test)
print "Precision: ", precision


recall = m.recall_score(pred, labels_test)
print "Recall: ", recall

f1_score= m.f1_score(pred, labels_test)
print "F1-Score: ", f1_score


from sklearn.metrics import classification_report

target_names = ["Not POI", "POI"]
print "Classification Report:"
print classification_report(y_true=labels_test, y_pred=pred, target_names=target_names)
"""
Classification Report:
             precision    recall  f1-score   support

    Not POI       0.89      1.00      0.94        33
        POI       1.00      0.20      0.33         5

avg / total       0.91      0.89      0.86        38
"""

"""
Classification Report:
             precision    recall  f1-score   support

    Not POI       0.91      0.94      0.93        33
        POI       0.50      0.40      0.44         5

avg / total       0.86      0.87      0.86        38
"""



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# --- stratified shuffle split cross validation
#from sklearn.model_selection import StratifiedShuffleSplit
#cv = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=42)
#
#from sklearn.model_selection import GridSearchCV
#param_grid = {
#         'C': [0.1, 1, 10, 100, 1000.],
#          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
#          }
#from sklearn.svm import SVC
#clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, cv = cv)
#clf = clf.fit(features_train,labels_train)
#print "Best estimator found by grid search:"
#print clf.best_estimator_

""" Result of best Classifier:
SVC(C=100, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.01, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
"""

#clf = AdaBoostClassifier(random_state=1)
#parameters = {"n_estimators": range(1,100), "random_state": 42}
#
#grid_obj = GridSearchCV(estimator=clf,param_grid=parameters)
#grid_fit =grid_obj.fit(features_train,labels_train)
#best_clf = grid_fit.best_estimator_
#
#predictions = (clf.fit(features_train, labels_train)).predict(features_test)
#accuracy = accuracy_score(predictions, labels_test)
### Report
#print best_clf
#print accuracy
#0.837209302326

""" Result of best Classifier:
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=1, random_state=42)
"""

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)