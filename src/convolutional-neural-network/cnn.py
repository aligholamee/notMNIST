# ========================================
# [] File Name : cnn.py
#
# [] Creation Date : January 2018
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================

"""
    Traicdning and Validation on notMNIST Dataset
    Implementation of a convolutional neural network with tensorflow on the notMNIST dataset.
"""
import numpy as np
import tensorflow as tf
import sys
import math
import pickle as pickle

# Data destination path
PICKLE_FILE = "../../data/notMNIST.pickle"

# Load the data to the RAM
with open(PICKLE_FILE, 'rb') as f:
    SAVE_FILE = pickle.load(f)

    """
        All of the following will be loaded in the form of a numpy array
    """

    TRAIN_DATASET = SAVE_FILE['train_dataset']
    TRAIN_LABELS = SAVE_FILE['train_labels']

    VALID_DATASET = SAVE_FILE['valid_dataset']
    VALID_LABELS = SAVE_FILE['valid_labels']

    TEST_DATASET = SAVE_FILE['test_dataset']
    TEST_LABELS = SAVE_FILE['test_labels']

    # Free some memory 
    del SAVE_FILE

    # Display the loaded data
    print("Training set: ", TRAIN_DATASET.shape, TRAIN_LABELS.shape)
    print("Validation set: ", VALID_DATASET.shape, VALID_LABELS.shape)
    print("Test set :", TEST_DATASET.shape, TEST_LABELS.shape)

DATASETS = {
   "IMAGE_SIZE": 28, 
   "NUM_LABELS": 10,
   "NUM_CHANNELS": 10
}

DATASETS["TOTAL_IMAGE_SIZE"] = DATASETS["IMAGE_SIZE"] * DATASETS["IMAGE_SIZE"]

def accuracy(predictions, labels):
    """
        Divides the number of true predictions to the number of total predictions
    """
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def reformat(dataset, labels, name):
    """
        Reformat the data to the one-hot and flattened mode
    """

    dataset = dataset.reshape((-1, DATASETS["IMAGE_SIZE"], DATASETS["IMAGE_SIZE"], DATASETS["NUM_CHANNELS"])).astype(np.float32)
    labels = (np.arange(DATASETS["NUM_LABELS"]) == labels[:, None]).astype(np.float32)

    print(name, " set", dataset.shape, labels.shape)

    return dataset, labels

DATASETS["TRAIN"], DATASETS["TRAIN_LABELS"] = reformat(TRAIN_DATASET, TRAIN_LABELS, "Train")
DATASETS["TEST"], DATASETS["TEST_LABELS"] = reformat(TEST_DATASET, TEST_LABELS, "Test")
DATASETS["VALID"], DATASETS["VALID_LABELS"] = reformat(VALID_DATASET, VALID_LABELS, "Valid")


print(DATASETS.keys())

