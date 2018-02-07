# ========================================
# [] File Name : learning_rate_decay.py
#
# [] Creation Date : January 2018
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================

"""
    Training and Validation on notMNIST Dataset.
    Implementing some regularization techniques on various classification methods.
    Regularization improves the final test accuarcy.
    This is the implementation of the learning rate decay regularization on the neural network.
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

def setup_neural_optimal(batch_size, rate_alpha, l2_beta, hidden_sizes, data, dropout_rate, decay_rate, decay_steps):
    """
        Implements a multilayer neural network with different hidden sizes 
        It also adds the dropout and learning rate regularization techniques in the computational graph.
    """
    graph = tf.Graph()

    with graph.as_default():

        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train = tf.placeholder(tf.float32, shape=(batch_size, data["TOTAL_IMAGE_SIZE"]))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, data["NUM_LABELS"]))
        tf_valid = tf.constant(data["valid"])
        tf_test = tf.constant(data["test"])

        hidden_loss = None

        prev_size = data["TOTAL_IMAGE_SIZE"]
        train_hidden = tf_train
        valid_hidden = tf_valid
        test_hidden = tf_test

        for index, size in enumerate(hidden_sizes):
            with tf.name_scope("hidden" + str(index + 1)):
                print("Hidden layer", index + 1, "with size", size, "from", prev_size)
                hidden_weights = tf.Variable(tf.truncated_normal([prev_size, size], stddev=math.sqrt(2.0/prev_size), seed=10001 + index))
                hidden_biases = tf.Variable(tf.zeros([size]))

                train_hidden = tf.nn.relu(tf.matmul(train_hidden, hidden_weights) + hidden_biases)
                valid_hidden = tf.nn.relu(tf.matmul(valid_hidden, hidden_weights) + hidden_biases)
                test_hidden = tf.nn.relu(tf.matmul(test_hidden, hidden_weights) + hidden_biases)

                hidden_dropout = tf.nn.dropout(train_hidden, dropout_rate, seed=20202 + index)
                
                hidden_l2 = tf.nn.l2_loss(hidden_weights)
                hidden_loss = hidden_l2 + hidden_loss if hidden_loss is not None else hidden_l2

            # Update the previous size
            prev_size = size
        
        with tf.name_scope("output"):

            weights = tf.Variable(
                tf.truncated_normal([prev_size, data["NUM_LABELS"]], stddev=math.sqrt(2.0/prev_size), seed=33333)
            )
            biases = tf.Variable(tf.zeros([data["NUM_LABELS"]]))

            train_logits = tf.matmul(hidden_dropout, weights) + biases
            valid_logits = tf.matmul(valid_hidden, weights) + biases
            test_logits = tf.matmul(test_hidden, weights) + biases

        # Training computation.
        cross = tf.nn.softmax_cross_entropy_with_logits(logits=train_logits, labels=tf_train_labels)
        loss = tf.reduce_mean(cross)
        loss += l2_beta * (hidden_loss + tf.nn.l2_loss(weights))
        
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(rate_alpha, global_step, decay_steps, decay_rate, staircase=True)

        info = {
            "GRAPH": graph,
            "BATCH_SIZE": batch_size,
            "TF_TRAIN": tf_train,
            "TF_TRAIN_LABELS": tf_train_labels,
            "LOSS": loss,
            # Optimizer.
            "OPTIMIZER": tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step),
            # Predictions for the training, validation, and test data.
            "TRAIN": tf.nn.softmax(train_logits),
            "VALID": tf.nn.softmax(valid_logits),
            "TEST": tf.nn.softmax(test_logits)
        }
    return info


OPTIMAL_STEPS = 10000
optimal_graph = setup_neural_optimal(batch_size=128, hidden_sizes=[1024, 1024, 512, 64]
                                    ,rate_alpha=0.5, l2_beta=0.001
                                    ,data=DATASETS, dropout_rate=0.5
                                    , decay_steps=OPTIMAL_STEPS, decay_rate=0.90)
run_graph(optimal_graph, DATASETS, 3000)
