# ========================================
# [] File Name : neural_net.py
#
# [] Creation Date : January 2018
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================

"""
    Training and Validation on notMNIST Dataset
    A neural network with 1 layer of 1024 hidden nodes implemented using tensorflow
    Stochastic gradient descent is also used as the optimizer
"""

import pickle as pickle
import numpy as np
import tensorflow as tf

# Data destination path
PICKLE_FILE = "../../../data/notMNIST.pickle"

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

IMAGE_SIZE = 28
NUM_LABELS = 10

def accuracy(predictions, labels):
    """
        Divides the number of true predictions to the number of total predictions
    """
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def reformat(dataset, labels):
    """
        Reformat the data to the one-hot and flattened mode
    """
    n_dataset = dataset.reshape((-1, IMAGE_SIZE * IMAGE_SIZE)).astype(np.float32)

    # Convert to the one hot format
    n_labels = (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)

    return n_dataset, n_labels

TRAIN_DATASET, TRAIN_LABELS = reformat(TRAIN_DATASET, TRAIN_LABELS)
VALID_DATASET, VALID_LABELS = reformat(VALID_DATASET, VALID_LABELS)
TEST_DATASET, TEST_LABELS = reformat(TEST_DATASET, TEST_LABELS)

# Display the openend files
print("Training Set ", TRAIN_DATASET.shape, TRAIN_LABELS.shape)
print("Validation Set", VALID_DATASET.shape, VALID_LABELS.shape)
print("Test Set", TEST_DATASET.shape, TEST_LABELS.shape)

# Implements a gradient descent using tensorflow computational graph
BATCH_SIZE = 128
HIDDEN_NODES = 1024

GRAPH = tf.Graph()

with GRAPH.as_default():
    """
        For the training data we use place holders in order to feed them
        in the run time with those mini bitches :D
    """
    TF_TRAIN_DATASET = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE))
    TF_TRAIN_LABELS = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
    TF_VALID_DATASET = tf.constant(VALID_DATASET)
    TF_TEST_DATASET = tf.constant(TEST_DATASET)

    """
       A nice hidden layer with 1024 nodes
    """
    with tf.name_scope("MehradHidden"):
        """
            Initialize the hidden weights and biases
        """
        HIDDEN_WEIGHTS = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, HIDDEN_NODES]))
        HIDDEN_BIASES = tf.Variable(tf.zeros([HIDDEN_NODES]))

        """ 
            Compute the logits WX + b and then apply D(S(WX + b), L) on them for the hidden layer
            The relu is applied on the hidden layer nodes only
        
        """
        TRAIN_HIDDEN_LOGITS = tf.nn.relu(tf.matmul(TF_TRAIN_DATASET, HIDDEN_WEIGHTS) + HIDDEN_BIASES)
        VALID_HIDDEN_LOGITS = tf.nn.relu(tf.matmul(TF_VALID_DATASET, HIDDEN_WEIGHTS) + HIDDEN_BIASES)
        TEST_HIDDEN_LOGITS = tf.nn.relu(tf.matmul(TF_TEST_DATASET, HIDDEN_WEIGHTS) + HIDDEN_BIASES)

    with tf.name_scope("Softmax-Linear"):
        """
            Initialize the main weights and biases
        """
        WEIGHTS = tf.Variable(tf.truncated_normal([HIDDEN_NODES, NUM_LABELS]))
        BIASES = tf.Variable(tf.zeros([NUM_LABELS]))

        """
            Compute the logits WX + b and the apply D(S(WX + b), L) on them for the final layer
        """
        TRAIN_LOGITS = tf.matmul(TRAIN_HIDDEN_LOGITS, WEIGHTS) + BIASES
        VALID_LOGTIS = tf.matmul(VALID_HIDDEN_LOGITS, WEIGHTS) + BIASES
        TEST_LOGITS = tf.matmul(TEST_HIDDEN_LOGITS, WEIGHTS) + BIASES

        LOSS = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=TRAIN_LOGITS, labels=TF_TRAIN_LABELS))

        OPTIMIZER = tf.train.GradientDescentOptimizer(0.5).minimize(LOSS)

        TRAIN_PREDICTION = tf.nn.softmax(TRAIN_LOGITS) 
        VALID_PREDICTION = tf.nn.softmax(VALID_LOGTIS)
        TEST_PREDICTION = tf.nn.softmax(TEST_LOGITS)

NUM_ITERATIONS = 3001

with tf.Session(graph=GRAPH) as session:
    """
        Start the above variable initialization
    """
    tf.initialize_all_variables().run()
    print("Variables initialized")

    for step in range(NUM_ITERATIONS):
        """
            Generate a random base and then generate a minibatch
        """
        BASE = (step * BATCH_SIZE) % (TRAIN_LABELS.shape[0] - BATCH_SIZE)
        BATCH_DATA = TRAIN_DATASET[BASE:(BASE + BATCH_SIZE), :]
        BATCH_LABELS = TRAIN_LABELS[BASE:(BASE + BATCH_SIZE), :]
        """
            Feed the current session with batch data
        """
        FEED_DICT = {TF_TRAIN_DATASET: BATCH_DATA, TF_TRAIN_LABELS: BATCH_LABELS}
        _, l, predictions = session.run([OPTIMIZER, LOSS, TRAIN_PREDICTION], feed_dict=FEED_DICT)
        
        if(step % 500 == 0):
            print("Minibatch loss at step ", step, ": ", l)
            print("Minibatch accuracy: ", accuracy(predictions, BATCH_LABELS))
            print("Validation accuracy: ", accuracy(VALID_PREDICTION.eval(), VALID_LABELS))
    """
        Displays the test prediction results
    """
    print("Test accuracy: ", accuracy(TEST_PREDICTION.eval(), TEST_LABELS))
