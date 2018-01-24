# ========================================
# [] File Name : model.py
#
# [] Creation Date : January 2018
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================

"""
    Training and Validation on notMNIST Dataset
    Fully connected network implementation with tensorflow
"""
import pickle as pickle
import numpy as np
import tensorflow as tf

# Data destination path
PICKLE_FILE = "../../data/notMNIST.pickle"

# Load the data to the RAM
with open(PICKLE_FILE, "rb") as f:
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
TRAIN_SUBSET = 10000

GRAPH = tf.Graph()

with GRAPH.as_default():

    """
        Load the training, validation and test data into the constants attached to the graph
    """
    TF_TRAIN_DATASET = tf.constant(TRAIN_DATASET[:TRAIN_SUBSET, :])
    TF_TRAIN_LABELS = tf.constant(TRAIN_LABELS[:TRAIN_SUBSET])
    TF_VALID_DATASET = tf.constant(VALID_DATASET[:TRAIN_SUBSET])
    TF_TEST_DATASET = tf.constant(TEST_DATASET[:TRAIN_SUBSET])


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

NUM_ITERATIONS = 3000

with tf.Session(graph=GRAPH) as session:
    """
        Start the above variable initialization
    """
    tf.initialize_all_variables().run()
    print("Variables initialized")

    for step in range(NUM_ITERATIONS):
            _, l, predictions = session.run([OPTIMIZER, LOSS, TRAIN_PREDICTION])
            if(step % 100 == 0):
                print("Loss at step ", step, ": ", l)
                print("Training accuracy: ", accuracy(predictions, TRAIN_LABELS[:TRAIN_SUBSET, :]))
    
    """
        Displays the test prediction results
    """
    print("Validation accuracy: ", accuracy(VALID_PREDICTION.eval(), VALID_LABELS))
    print("Test accuracy: ", accuracy(TEST_PREDICTION.eval(), TEST_LABELS))



    

    


