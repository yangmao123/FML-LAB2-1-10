# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 08:24:23 2020

@author: 75965
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

FILE_PATH = 'tic-tac-toe.txt'
df = pd.read_csv(FILE_PATH)
source = np.array(df)
print(source)
NumDataPerClass=200
X = source[:,[0,1,2,3,4,5,6,7,8]]
y = source[:,[9]]
rIndex = np.random.permutation(2*NumDataPerClass)
Xr = X[rIndex,]
yr = y[rIndex]
# Training and test sets (half half)
#
X_train = Xr[0:NumDataPerClass]
y_train = yr[0:NumDataPerClass]
X_test = Xr[NumDataPerClass:2*NumDataPerClass]
y_test = yr[NumDataPerClass:2*NumDataPerClass]
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
Ntrain = NumDataPerClass;
Ntest = NumDataPerClass;
w = np.random.randn(9)
print(w)

def PercentCorrect(Inputs, targets, weights):
    N = len(targets)
    nCorrect = 0
    for n in range(N):
        OneInput = Inputs[n,:]
        if (targets[n] * np.dot(OneInput, weights) > 0):
            nCorrect +=1
    return 100*nCorrect/N


print('Initial Percentage Correct: %6.2f' %(PercentCorrect(X_train, y_train, w)))
MaxIter=2500
alpha = 0.002
# Space to save answers for plotting
#
P_train = np.zeros(MaxIter)
P_test = np.zeros(MaxIter)
# Main Loop
#
for iter in range(MaxIter):
# Select a data item at random
#
    r = np.floor(np.random.rand()*Ntrain).astype(int)
    x = X_train[r,:]
# If it is misclassified, update weights
#
    if (y_train[r] * np.dot(x, w) < 0):
        w += alpha * y_train[r] * x
# Evaluate trainign and test performances for plotting
#
    P_train[iter] = PercentCorrect(X_train, y_train, w);
    P_test[iter] = PercentCorrect(X_test, y_test, w);
print('Percentage Correct After Training: %6.2f %6.2f'
      %(PercentCorrect(X_train, y_train, w), PercentCorrect(X_test, y_test, w)))

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(range(MaxIter), P_train, 'b', label = "Training")
ax.plot(range(MaxIter), P_test, 'r', label = "Test")
ax.grid(True)
ax.legend()
ax.set_title('Perceptron Learning')
ax.set_ylabel('Training and Test Accuracies', fontsize=14)
ax.set_xlabel('Iteration', fontsize=14)
plt.savefig('learningCurves.png')