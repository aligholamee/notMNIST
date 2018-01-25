# ========================================
# [] File Name : l2_regularization.py
#
# [] Creation Date : January 2018
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================

"""
    Training and Validation on notMNIST Dataset.
    Implementing some regularization techniques on the training methods we talked before.
    Improves the final test accuarcy.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import math
import pickle as pickle

# Data destination path
PICKLE_FILE = "../../data/notMNIST.pickle"

# Load the data to the RAM
with open(PICKLE_FILE, 'rb') as f:
    SAVE_FILE = pickle.load(f)

    TRAIN_DATASET = SAVE_FILE['train_dataset']
    TRAIN_LABELS = SAVE_FILE['train_labels']

    VALID_DATASET = SAVE_FILE['valid_dataset']
    VALID_LABELS = SAVE_FILE['valid_labels']

    TEST_DATASET = SAVE_FILE['test_dataset']
    TEST_LABELS = SAVE_FILE['test_labels']

    # Free some memory
    del SAVE_FILE


    print("Training set: ", TRAIN_DATASET.shape, TRAIN_LABELS.shape)
    print("Validation set: ", VALID_DATASET.shape, VALID_LABELS.shape)
    print("Test set: ", TEST_DATASET.shape, TEST_LABELS.shape)

DATASETS = {
    "IMAGE_SIZE": 28,
    "NUM_LABELS": 10
}

DATASETS["TOTAL_IMAGE_SIZE"] = DATASETS["IMAGE_SIZE"] * DATASETS["IMAGE_SIZE"]

def reformat_dataset(dataset, labels, name):
    """
        Reformat the data to the one-hot and flattened mode
    """
    dataset = dataset.reshape((-1, DATASETS["TOTAL_IMAGE_SIZE"])).astype(np.float32)
    labels = (np.arange(DATASETS["NUM_LABELS"] == labels[:, None])).astype(np.float32)
    print(name + " set", dataset.shape, labels.shape)

    return dataset, labels

DATASETS["train"], DATASETS["train_labels"] = reformat(TRAIN_DATASET, TRAIN_LABELS, "Training")
DATASETS["valid"], DATASETS["valid_labels"] = reformat(VALID_DATASET, VALID_LABELS, "Validation")
DATASETS["test"], DATASETS["test_labels"] = reformat(TEST_DATASET, TEST_LABELS, "Test")

print(DATASETS.keys())

def accuracy(predictions, labels):
    """
        Divides the number of true predictions to the number of total predictions
    """
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def run_graph(graph_info, data, step_count):
    """
        Initializes and runs the tensor's graph
    """
    with tf.Session(graph = graph_info["GRAPH"]) as session:
        tf.initialize_all_variables().run()
        print("Initialized!")

        BATCH_SIZE = graph_info["BATCH_SIZE"]

        for step in range(step_count + 1):
            BASE = (step * BATCH_SIZE) % (data["train_labels"].shape[0] - BATCH_SIZE)
            BATCH_DATA = data["train"][BASE:(BASE + BATCH_SIZE), :]
            BATCH_LABELS = data["train_labels"][BASE:(BASE + BATCH_SIZE), :]

            TARGETS = [graph_info["OPTIMIZER"], graph_info["LOSS"], graph_info["train"]]

            FEED_DICT = {graph_info["TF_TRAIN_DATASET"]: BATCH_DATA, graph_info["TF_TRAIN_LABELS"]: BATCH_LABELS}

            _, l, predictions = session.run(TARGETS, feed_dict=FEED_DICT)
            if(step % 500 == 0):
                print("Minibatch loss at step ", step, ":", l)
                print("Minibatch accuracy: ", accuracy(predictions, BATCH_LABELS))
                print("Validation accuracy: ", accuracy(graph_info["VALID"].eval(), data["VALID_LABELS"]))

        print("Test accuracy: ", accuracy(graph_info["TEST"].eval(), data["TEST_LABELS"]))
        
