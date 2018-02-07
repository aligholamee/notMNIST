# notMNIST
<p align="center">
    <img src="http://yaroslavvb.com/upload/notMNIST/nmn.png">
</p>

---
Implementation of multiple machine learning classifiers and regularization techniques on the notMNIST dataset.

## Starting Point
Feel free to have a look at the code. Feel more free to **contribute**. First of all, you need to head into the _logistic-regression_ folder and run the _logistic classifier_ (model.py). The result will be:
+ A downloaded dataset.
+ A brief introduction on the dataset dimensionalities and characteristics.

_Now you are good to go_. Play with the code as much as you can and run it to see the results.

## Suggested Platform
You won't suffer from running the _logistic_ classifier. But, you can obviously see the performance defficiency while running the _CNN_ classifier on _CPU_. Its suggested to use the _tensorflow-gpu_. If you are on _ubuntu_ make sure you head to [this tutorial](https://github.com/williamFalcon/tensorflow-gpu-install-ubuntu-16.04) to setup the environment for the deep leaning with _tensorflow-gpu_.

## Not Another MNIST Classification Task
This is not another classification task on the MNIST dataset because of the two following reasons.
+ The dataset is more complicated. So, there was a vast need for data preprocessing.
+ Hyperparameters are not left alone. Various tweaks like _L1_ and _L2_ _regularization_, _dropout_, _learning rate decay_ are applied to improve the accuracy.

## Implemented Classifiers
Below is the list of implemented classifiers using different regularization techniques and the respective accuracy in each dataset category.


| Classifier        | Regularization           | Training Accuracy  | Validation Accuracy | Test Accuracy | Optimizer |
| ------------- |:-------------:| :-----:| :-------------: |:-------------:|:-------------:|
| Single-Layer Perceptron     | - | 78.6 |75.82      | 82.26 | Gradient Descent |
| Two-Layer Perceptron with 1024 Hidden Nodes     | - |   78.90 (Batch of 128) |80.01      | 85.92 |Gradient Descent|
| Single-Layer Perceptron     | - | 74.21 (Batch of 128) |76.17     | 83.1 | Stochastic Gradient Descent |
| Single-Layer Perceptron     | L2 | 84.375 (Batch of 128) |86.66     | 92.67 | Stochastic Gradient Descent |
| Two-Layer Network with 1024 Hidden Nodes     | Dropout |   85.93 (Batch of 128) |86.9      | 92.89 |Gradient Descent|
| Logistic Regression     | L2 |   78.125 |81.1      | 87.8 |Gradient Descent|
| Five-Layer Network with 1024, 1024, 512, 64 Hidden Nodes     | Learning Rate Decay |   87.5 (Batch of 128) |87.72     | 93.11 |Gradient Descent|

## Special Thanks
First of all, thank you **Tensorflow**. Secondly, I have to thank **Udacity** community for teaching me the principle terms of building networks and tuning hyperparameters and finally I have to say thanks to the Dr. Vincent Vanhoucke at Google for his great explanation of deep learning course(UD730).
