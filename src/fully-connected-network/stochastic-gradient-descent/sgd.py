# ========================================
# [] File Name : sgd.py
#
# [] Creation Date : January 2018
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================

"""
    Training and Validation on notMNIST Dataset
    Stochastic gradient descent implementation with tensorflow
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
        Reformat data to the one-hot and flattened mode
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

GRAPH = tf.Graph()

with GRAPH.as_default():

    """
        Create place holders for the training data shape, with respect to the batch size
    """
    TF_TRAIN_DATASET = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE))
    TF_TRAIN_LABELS = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
    TF_VALID_DATASET = tf.constant(VALID_DATASET)
    TF_TEST_DATASET = tf.constant(TEST_DATASET)


    """
        Initialize the weights matrix with normal distribution and the biases with zero values
    """
    WEIGHTS = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, NUM_LABELS]))
    BIASES = tf.Variable(tf.zeros([NUM_LABELS]))

    """
        Compute the logits WX + b and then apply D(S(WX + b), L) on them
    """
    LOGITS = tf.matmul(TF_TRAIN_DATASET, WEIGHTS) + BIASES
    LOSS = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = TF_TRAIN_LABELS, logits = LOGITS))

    """
        Find the minimum of the loss using gradient descent optimizer
        remember that the optimizer is an algorithm now - ready to be tested on the test data
    """
    OPTIMIZER = tf.train.GradientDescentOptimizer(0.1).minimize(LOSS)

    """
        Predictions for the training, validation, and test data.
    """
    TRAIN_PREDICTION = tf.nn.softmax(LOGITS)
    VALID_PREDICTION = tf.nn.softmax(tf.matmul(TF_VALID_DATASET, WEIGHTS) + BIASES)
    TEST_PREDICTION = tf.nn.softmax(tf.matmul(TF_TEST_DATASET, WEIGHTS) + BIASES)

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
