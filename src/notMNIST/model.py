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
from numpy import random
import config

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


# ======================================== #
# =========== Load the letter ============ #
# ======================================== #
def loadLetter(folder, minNumOfImages):
    """Load the data for a single letter label."""

    imageFiles = os.listdir(folder)
    dataset = np.ndarray(shape=(len(imageFiles), config.imageSize, config.imageSize),
                         dtype=np.float32)
    print(folder)

    for imageIndex, image in enumerate(imageFiles):
        imageFile = os.path.join(folder, image)
        try:
            imageData = (ndimage.imread(imageFile).astype(float) - 
                    config.pixelDepth / 2) / config.pixelDepth
        if imageData.shape != (imageSize, imageSize):
            raise Exception('Unexpected image shape: %s' % str(imageData.shape))
        dataset[imageIndex, :, :] = imageData

        except IOError as e:
            print('Could not read:', imageFile, ':', e, '- it\'s ok, skipping.')
    
    numImages = imageIndex + 1
    dataset = dataset[0:numImages, :, :]
    if numImages < minNumOfImages:
        raise Exception('Many fewer images than expected: %d < %d' % (numImages, minNumOfImages))
    
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))

    return dataset

# ======================================== #
# ========== Pickle the data  ============ #
# ======================================== #
def pickleData(dataFolders, minNumOfImagesPerClass, force=False):
    datasetNames = []

    # Create an array of pickled files in dataset
    for folder in dataFolders: 
        setFileName = folder + '.pickle'
        datasetNames.append(setFileName)

        # In case the data is pickled already
        if os.path.exists(setFileName) and not force:
            print('%s has already pickled. Skipping... ' %setFileName)
        else:
            print('Pickling %s.' % setFileName)
            dataset = loadLetter(folder, minNumOfImagesPerClass)
        try:
            with open(setFileName, 'wb') as f:
                pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', setFileName, ':', e)
        
    return datasetNames


testFileName = getData('notMNIST_small.tar.gz', 8458043)
# trainFileName = getData('notMNIST_large.tar.gz', 247336696)

testFolders = extractData(testFileName)
# trainFolders = extractData(trainFileName)



