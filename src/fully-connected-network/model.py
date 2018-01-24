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
    save = pickle.load(f)

    train_dataset = save['train_dataset'] 
    train_labels = save['train_labels']

    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']

    test_dataset = save['test_dataset']
    test_labels = save['test_labels']

    # Free some memory
    del save

# Reformat to the one-hot encoding mode
# def reformatData(dataset, labels):
image_size = 28
num_labels = 10
def reformat(dataset, labeles):
    n_dataset = dataset.reshape((1, image_size * image_size)).astype(np.float)

    # Convert to the one hot format
    n_labels = (np.arange(num_labels) == n_dataset[:, None]).astype(np.float)

    return n_dataset, n_labels

train_dataset, train_labels =  reformat(train_dataset, train_labels) 
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

# Display the openend files 
print("Training Set ", train_dataset.shape, train_labels.shape)
print("Validation Set", valid_dataset.shape, valid_labels.shape)
print("Test Set", test_dataset.shape, test_labels.shape)

# Implements a gradient descent using tensorflow computational graph
train_subset = 10000

