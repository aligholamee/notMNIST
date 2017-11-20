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
num_classes = 10
np.random.seed(133)

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

# ======================================== #
# ========== Extract the data ============ #
# ======================================== #
def extractData(fileName, force=False):
    root = os.path.splitext(os.path.splitext(fileName)[0])[0]  # remove .tar.gz

    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, fileName))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(fileName)
        sys.stdout.flush()
        tar.extractall(dataRoot)
        tar.close()
    data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]

    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)

    return data_folders


testFileName = getData('notMNIST_small.tar.gz', 8458043)
trainFileName = getData('notMNIST_large.tar.gz', 247336696)

testFolders = extractData(testFileName)
trainFolders = extractData(trainFileName)