# Training and Validation on notMNIST Dataset
# Fully connected network implementation with tensorflow
# ========================================
# [] File Name : model.py
#
# [] Creation Date : January 2018
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================
import pickle as pickle
import numpy as np 
import tensorflow as tf

# Data destination path
pickle_file = "../../data/notMNIST.pickle"

# Load the data to the RAM
with open(pickle_file, "rb") as f:
    SAVE_FILE = pickle.load(f)

    TRAIN_DATASET = SAVE_FILE['train_dataset'] 
    TRAIN_LABELS = SAVE_FILE['train_labels']

    VALID_DATASET = SAVE_FILE['valid_dataset']
    VALID_LABELS = SAVE_FILE['valid_labels']

    TEST_DATASET = SAVE_FILE['test_dataset']
    TEST_LABELS = SAVE_FILE['test_labels']

    # Free some memory
    del SAVE_FILE

# Reformat to the one-hot encoding mode
# def reformatData(dataset, labels):
image_size = 28
num_labels = 10
def reformat(dataset, labels):
    n_dataset = dataset.reshape((1, image_size * image_size)).astype(np.float)

    # Convert to the one hot format
    n_labels = (np.arange(num_labels) == labels[:, None]).astype(np.float)

    return n_dataset, n_labels

TRAIN_DATASET, TRAIN_LABELS =  reformat(TRAIN_DATASET, TRAIN_LABELS) 
VALID_DATASET, VALID_LABELS = reformat(VALID_DATASET, VALID_LABELS)
TEST_DATASET, TEST_LABELS = reformat(TEST_DATASET, TEST_LABELS)

# Display the openend files 
print("Training Set ", TRAIN_DATASET.shape, TRAIN_LABELS.shape)
print("Validation Set", VALID_DATASET.shape, VALID_LABELS.shape)
print("Test Set", TEST_DATASET.shape, TEST_LABELS.shape)

# Implements a gradient descent using tensorflow computational graph
train_subset = 10000

