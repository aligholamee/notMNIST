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

# Constants 
url = 'https://commondatastorage.googleapis.com/books1000/'
lastPercentReported = None
dataRoot = '.'

# ======================================== #
# ============= Progress Hook ============ #
# ======================================== #
def downloadProgressHook(count, blockSize, totalSize):
    # Reports the progress of a download
    global lastPercentReported
    percent = int(count * blockSize * 100 / totalSize)

    if lastPercentReported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        lastPercentReported = percent

# ======================================== #
# ========== Download The Data =========== #
# ======================================== #
def getData(filename, expectedBytes, force = False):
    # Download a file if not present, and make sure it's the right size
    destinationFileName = os.path.join(dataRoot, filename)

    # Get the file
    if force or not os.path.exists(destinationFileName):
        print('Attempting to download:', filename) 
        filename, _ = urlretrieve(url + filename, destinationFileName, reporthook = downloadProgressHook)
        print('\nDownload Complete!')

    statinfo = os.stat(destinationFileName)

    # Verify the download
    if statinfo.st_size == expectedBytes:
        print('Found and verified', destinationFileName)
    else:
        raise Exception(
            'Failed to verify ' + destinationFileName + '. Can you get to it with a browser?')

    return destinationFileName

test_filename = getData('notMNIST_small.tar.gz', 8458043)
train_filename = getData('notMNIST_large.tar.gz', 247336696)
