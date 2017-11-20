# Softmax Function Implementation
# ========================================
# [] File Name : softmax.py
#
# [] Creation Date : December 2017
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================
#
import numpy as np 
import matplotlib.pyplot as plt

scores = [3.0, 1.0, 0.2]


def softmax(x):
    # Compute softmax values for each sets of scores in x.
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Plot softmax curves
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()










