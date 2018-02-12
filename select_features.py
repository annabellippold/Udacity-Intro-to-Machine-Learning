# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 15:26:02 2018

@author: LIPPA2
"""

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

from sklearn.feature_selection import SelectKBest
from pprint import pprint

# --- Choose SelectKBest to get the features with the hights score value
#     Selecting K-Best because it's not clear which Features will be good features at this point.
def Select_K_Best(data_dict, features_list, k):
    """
    This function run the SelectKBest feature selection algotihm and returns 
    an array with the features and there score. Only the features with the highest
    scores will be used in poi_id.py
    """
    
    k_best_array = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(k_best_array)
    
    # create selector
    k_best = SelectKBest(k=k)
    # fit selector to data
    k_best.fit(features, labels)
    # calculate scores, for ranking which feature is important
    scores = k_best.scores_
    # create a tuple that returns feature and score
    tuples = zip(features_list[1:], scores)
    # sort the tuble, that shows the feature with the highest score first
    k_best_features = sorted(tuples, key=lambda x: x[1], reverse=True)
    
    return k_best_features[:k]

# --- Find features with the highest Score
# Given features (financial, email and poi):
financial_features = ['salary', 'deferral_payments', 'total_payments', 
                      'loan_advances', 'bonus', 'restricted_stock_deferred', 
                      'deferred_income', 'total_stock_value', 'expenses', 
                      'exercised_stock_options', 'other', 'long_term_incentive', 
                      'restricted_stock', 'director_fees']

email_features = ['to_messages', 'from_poi_to_this_person', 
                  'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

new_features = ['bonus_ratio', 'poi_email_ratio']

poi_label = ['poi']

# --- Print out list with features and scores
features_list = poi_label + financial_features + email_features + new_features
print Select_K_Best(data_dict, features_list, k = 20)


# --- Output of all features and there scores
"""
[('exercised_stock_options', 25.097541528735491), 
('total_stock_value', 24.467654047526398), 
('bonus', 21.060001707536571), 
('salary', 18.575703268041785), 
('deferred_income', 11.595547659730601), 
('long_term_incentive', 10.072454529369441), 
('restricted_stock', 9.3467007910514877), 
('total_payments', 8.8667215371077752), 
('shared_receipt_with_poi', 8.7464855321290802), 
('loan_advances', 7.2427303965360181), 
('expenses', 6.2342011405067401), 
('from_poi_to_this_person', 5.3449415231473374), 
('other', 4.204970858301416), 
('from_this_person_to_poi', 2.4265081272428781), 
('director_fees', 2.1076559432760908), 
('to_messages', 1.6988243485808501), 
('deferral_payments', 0.2170589303395084), 
('from_messages', 0.16416449823428736), 
('restricted_stock_deferred', 0.06498431172371151)]
"""

#--- Test K-best with new features, which Score get them?
"""
[('poi_email_ratio', 16.484649768125919), 
('bonus_ratio', 10.914952845206287), 
('shared_receipt_with_poi', 8.9038215571655712), 
('from_poi_to_this_person', 5.4466874833253529), 
('loan_advances', 2.5182610445203437), 
('from_this_person_to_poi', 2.470521222656084), 
('to_messages', 1.7516942790340737), 
('director_fees', 0.54908420147980874), 
('total_payments', 0.34962715304280179), 
('deferral_payments', 0.23899588985313305), 
('exercised_stock_options', 0.22826733729104948), 
('deferred_income', 0.21950572394230994), 
('total_stock_value', 0.16611912320976671), 
('from_messages', 0.1587702392129193), 
('bonus', 0.077948855777229875), 
('other', 0.068194519159558625), 
('restricted_stock', 0.031333216297618496), 
('long_term_incentive', 0.022229270861607336), 
('expenses', 0.01397841382175243), 
('restricted_stock_deferred', 0.0041731922805086684)]
"""