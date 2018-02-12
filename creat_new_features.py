# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 17:47:33 2018

@author: LIPPA2
"""
import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


features_list = ['poi','exercised_stock_options', 'total_stock_value', 'bonus',
                 'salary', 'deferred_income', 'long_term_incentive',
                 'restricted_stock', 'total_payments', 'shared_receipt_with_poi',
                 'loan_advances', 'expenses', 'from_poi_to_this_person', 'from_this_person_to_poi']
# ---


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

def Poi_Email_Rate(data_dict, features_list):
    """
    The PoiEmailRatio calculates
    """
    features = ['from_messages', 'to_messages', 'from_poi_to_this_person',
                'from_this_person_to_poi']

    for key in data_dict:
        # Select Person from data_dict
        employee = data_dict[key]
        # create variable for checking NaN
        ok = True
        for feature in features:
            # check if feature is NaN then set ok to False
            if employee[feature] == 'NaN':
                ok = False
        if ok:
            # Calculate the total number of "from_mails" per Person
            from_total = employee['from_poi_to_this_person'] + employee['from_messages']
            # Calculate the total number of "to_mails" per Person
            to_total = employee['from_this_person_to_poi'] + employee['to_messages']
            # Calculate ratio for "from_mails"                   
            from_poi_ratio = float(employee['from_poi_to_this_person']) / from_total
            # Calculate ratio for "to_mails"
            to_poi_ratio = float(employee['from_this_person_to_poi']) / to_total
            # Calculate the poi_email_ratio
            employee['poi_email_ratio'] = to_poi_ratio + from_poi_ratio
        else:
            employee['poi_email_ratio'] = 'NaN'
    
    # add new created feature to list
    features_list.append('poi_email_ratio')
    return features_list
    
#print Poi_Email_Rate(data_dict, features_list)

def Bonus_Rate(data_dict, features_list):
    """
    The BonusRatio calculates how many percent is the bonus from the 
    salary. Is is a very high bonus or a normal bonus?
    """

    features = ['bonus', 'salary']

    for key in data_dict:
        # Select Person from data_dict
        employee = data_dict[key]
        # create variable for checking NaN
        ok = True
        for feature in features:
            # check if feature is NaN then set ok to False
            if employee[feature] == 'NaN':
                ok = False
        if ok:
            # Calculate bonus_ratio
            employee['bonus_ratio'] = float(employee['bonus']) / employee['salary']
        else:
            employee['bonus_ratio'] = 'NaN'
    
    # add new created feature to list
    features_list.append('bonus_ratio')
    return features_list
    
#print Bonus_Rate(data_dict, features_list)
# Test functions on Jeffrey Skilling
#print data_dict['SKILLING JEFFREY K']['bonus_ratio']
#print data_dict['SKILLING JEFFREY K']['poi_email_ratio']

""" Test Result - if code is working fine
#--- added 2 new features to the list
['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 
'deferred_income', 'long_term_incentive', 'restricted_stock', 'total_payments', 
'shared_receipt_with_poi', 'loan_advances', 'expenses', 'from_poi_to_this_person', 
'from_this_person_to_poi', 'poi_email_ratio', 'bonus_ratio']

#--- calculations 
5.03933380007
0.457183037284
"""
