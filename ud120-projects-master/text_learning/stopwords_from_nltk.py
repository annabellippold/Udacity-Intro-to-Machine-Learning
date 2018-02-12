# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:16:29 2018

@author: lippa2
"""

# Import NLTK
import nltk

# Download stopwords from NLTK 
nltk.download('all', halt_on_error=False)
#nltk.download('stopwords')
from nltk.corpus import stopwords

# Get a list of stopwords, parameter = english
sw = stopwords.words("english")

print sw[0]
# Print lenght of list with stopwords.
print len(sw)
""" Result: 179 """

