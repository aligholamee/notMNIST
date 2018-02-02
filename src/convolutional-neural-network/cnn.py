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

def run_graph(graph_info, data, step_count, report_every=50):
    """
        Initializes and runs the tensor's graph
    """
    with tf.Session(graph=graph_info["GRAPH"]) as session:
        tf.initialize_all_variables().run()
        print("Initialized!")

        batch_size = graph_info["BATCH_SIZE"]

        for step in range(step_count + 1):
            base = (step * batch_size) % (data["TRAIN_LABELS"].shape[0] - batch_size)

            # Generate a minibatch
            batch_data = data["TRAIN"][base:(base + batch_size), :, :, :]
            batch_labels = data["TRAIN_LABELS"][base:(base + batch_size), :]

            # Prepare the targets
            targets = [graph_info["OPTIMIZER"], graph_info["LOSS"], graph_info["PREDICTIONS"]]
            feed_dict = {graph_info["TRAIN"]: batch_data, graph_info["LABELS"]: batch_labels}

            _, l, predictions = session.run(targets, feed_dict=feed_dict)
            
            if(step % report_every == 0):
                print("Minibatch loss at step, ", step, ": ", l)
                print("Minibatch accuracy: ", accuracy(predictions, batch_labels))
                print("Validation accuracy: ", accuracy(graph_info["VALID"].eval(), DATASETS["VALID_LABELS"]))
            
        print("Test accuracy: ", accuracy(graph_info["TEST"].eval(), DATASETS["TEST_LABELS"]))

def two_layer_convnet(batch_size, patch_size, depth, hidden_size, data):
    
    # Some temp variables
    image_size = DATASETS["IMAGE_SIZE"]
    num_labels = DATASETS["NUM_LABELS"]
    num_channels = DATASETS["NUM_CHANNELS"]
    graph = tf.Graph()
    with graph.as_default():
        # Input data
        train = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        labels= tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        valid = tf.constant(DATASETS["VALID"])
        test  = tf.constant(DATASETS["TEST"])

        # Variables
        layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
        layer1_biases = tf.Variable(tf.zeros([depth]))

        layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

        layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, hidden_size], stddev=0.1))
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[hidden_size]))

        layer4_weights = tf.Variable(tf.truncated_normal([hidden_size, num_labels], stddev=0.1))
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

        # Create connections and create model
        def model(set):
            conv   = tf.nn.conv2d(set, layer1_weights, [1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            conv   = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer2_biases)
            shape  = hidden.get_shape().as_list()
            reshape= tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
            return tf.matmul(hidden, layer4_weights) + layer4_biases

        # Training computation
        logits = model(train)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        
        info = {
            "GRAPH": graph,
            "BATCH_SIZE": batch_size,
            "TRAIN": train,
            "LABELS": labels,
            "LOSS": loss,
            "OPTIMIZER": tf.train.GradientDescentOptimizer(0.05).minimize(loss),

            # Predictions for the training, validation, and test data.
            "PREDICTIONS": tf.nn.softmax(logits),
            "VALID": tf.nn.softmax(model(valid)),
            "TEST":  tf.nn.softmax(model(test))
        }
    return info

GRAPH_2CONV = two_layer_convnet(batch_size=16, patch_size=5, depth=16, hidden_size=64, data=DATASETS)

run_graph(GRAPH_2CONV, DATASETS, 1000)