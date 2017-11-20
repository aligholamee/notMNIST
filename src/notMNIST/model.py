# Training and Validation on notMNIST Dataset
# ========================================
# [] File Name : model.py
#
# [] Creation Date : December 2017
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================
#
from __future__ import print_function
import matplotlib.pyplot as plt 
import numpy as np 
import os
import sys 
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle 

