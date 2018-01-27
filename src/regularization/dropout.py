# ========================================
# [] File Name : dropout.py
#
# [] Creation Date : January 2018
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================

"""
    Training and Validation on notMNIST Dataset.
    Implementing some regularization techniques on various classification methods.
    Regularization improves the final test accuarcy.
    This is the implementation of the dropout on the neural network.
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
    labels = (np.arange(DATASETS["NUM_LABELS"]) == labels[:, None]).astype(np.float32)
    print(name + " set", dataset.shape, labels.shape)

    return dataset, labels

DATASETS["train"], DATASETS["train_labels"] = reformat_dataset(TRAIN_DATASET, TRAIN_LABELS, "Training")
DATASETS["valid"], DATASETS["valid_labels"] = reformat_dataset(VALID_DATASET, VALID_LABELS, "Validation")
DATASETS["test"], DATASETS["test_labels"] = reformat_dataset(TEST_DATASET, TEST_LABELS, "Test")

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

            TARGETS = [graph_info["OPTIMIZER"], graph_info["LOSS"], graph_info["TRAIN"]]

            FEED_DICT = {graph_info["TF_TRAIN"]: BATCH_DATA, graph_info["TF_TRAIN_LABELS"]: BATCH_LABELS}

            _, l, predictions = session.run(TARGETS, feed_dict=FEED_DICT)
            if(step % 500 == 0):
                print("Minibatch loss at step ", step, ":", l)
                print("Minibatch accuracy: ", accuracy(predictions, BATCH_LABELS))
                print("Validation accuracy: ", accuracy(graph_info["VALID"].eval(), data["valid_labels"]))

        print("Test accuracy: ", accuracy(graph_info["TEST"].eval(), data["test_labels"]))

def setup_neural_dropout(batch_size, rate_alpha, l2_beta, hidden_size, data, dropout_rate):
    """
        Initializes a neural network with tensorflow computational graph
    """
    graph = tf.Graph()

    with graph.as_default():

        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train = tf.placeholder(tf.float32, shape=(batch_size, data["TOTAL_IMAGE_SIZE"]))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, data["NUM_LABELS"]))
        tf_valid = tf.constant(data["valid"])
        tf_test = tf.constant(data["test"])

        # Hidden layer
        with tf.name_scope("hidden1"):
            hidden_weights = tf.Variable(tf.truncated_normal([data["TOTAL_IMAGE_SIZE"], hidden_size]))
            hidden_biases = tf.Variable(tf.zeros([hidden_size]))
            train_hidden = tf.nn.relu(tf.matmul(tf_train, hidden_weights) + hidden_biases)
            valid_hidden = tf.nn.relu(tf.matmul(tf_valid, hidden_weights) + hidden_biases)
            test_hidden = tf.nn.relu(tf.matmul(tf_test, hidden_weights) + hidden_biases)
            # Applies the dropout regularization with respect to the hidden layers of the network and the dropout_rate
            hidden_dropout = tf.nn.dropout(train_hidden, dropout_rate, seed=12345)

        with tf.name_scope("softmax_linear"):
            weights = tf.Variable(tf.truncated_normal([hidden_size, data["NUM_LABELS"]]))
            biases = tf.Variable(tf.zeros([data["NUM_LABELS"]]))
            train_logits = tf.matmul(train_hidden, weights) + biases
            valid_logits = tf.matmul(valid_hidden, weights) + biases
            test_logits = tf.matmul(test_hidden, weights) + biases

        # Training computation.
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train_logits, labels=tf_train_labels))
        loss += l2_beta * (tf.nn.l2_loss(hidden_weights) + tf.nn.l2_loss(weights))

        info = {
            "GRAPH": graph,
            "BATCH_SIZE": batch_size,
            "TF_TRAIN": tf_train,
            "TF_TRAIN_LABELS": tf_train_labels,
            "LOSS": loss,
            # Optimizer.
            "OPTIMIZER": tf.train.GradientDescentOptimizer(rate_alpha).minimize(loss),
            # Predictions for the training, validation, and test data.
            "TRAIN": tf.nn.softmax(train_logits),
            "VALID": tf.nn.softmax(valid_logits),
            "TEST": tf.nn.softmax(test_logits)
        }
    return info

dropout_graph = setup_neural_dropout(batch_size=128, hidden_size=1024, rate_alpha=0.5, l2_beta=0.001, data=DATASETS, dropout_rate=0.5)
run_graph(dropout_graph, DATASETS, 3000)
