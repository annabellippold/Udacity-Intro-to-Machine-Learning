# -*- coding: utf-8 -*-
"""
Created on Sat Jan 06 19:41:55 2018

@author: lippa2
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)

def fit_scaler_on_original_data(data_dict, feature):

    feature_not_NaN = []
    for person in data_dict.keys():
        if data_dict[person][feature] != "NaN":
            feature_not_NaN.append(data_dict[person][feature])
    
    scaler = MinMaxScaler()
    scaler.fit(np.array(feature_not_NaN, dtype = np.float).reshape(-1, 1))
    
    return scaler

feature_1 = "salary"
feature_2 = "exercised_stock_options"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list)

# Fit a scaler on the original data (So we don't get deceived by the "NaN" points)
feature_1_scaler = fit_scaler_on_original_data(data_dict, feature_1)
feature_2_scaler = fit_scaler_on_original_data(data_dict, feature_2)

# Rescale the data using the appropriate scaler
rescaled_data = data
rescaled_data[:, 1] = feature_1_scaler.transform(rescaled_data[:, 1].reshape(1, -1))
rescaled_data[:, 2] = feature_2_scaler.transform(rescaled_data[:, 2].reshape(1, -1))

poi, finance_features = targetFeatureSplit(rescaled_data)

kmeans_model = KMeans(n_clusters = 2)
pred = kmeans_model.fit_predict(rescaled_data)

try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters_with_feature_scaling.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"
    
    
print "rescaled salary (200000.0) is:", feature_1_scaler.transform([[200000.0]])[0][0]
print "rescaled exercised_stock_options (1000000.0) is:", feature_2_scaler.transform([[1000000.0]])[0][0]