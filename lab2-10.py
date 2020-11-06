# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 06:26:20 2020

@author: 75965
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
'''
birth_data=[]
age=np.zeros((500,2),dtype=int)
with open("a.csv","r",encoding="utf-8") as f:
    f_read =csv.reader(f)
    next(f_read)
    for v in f_read:
        print(v)
        birth_data.append(v)
#    print(birth_data)
age=[[x[0] for x in birth_data],[x[1] for x in birth_data]]
print(age)
'''

FILE_PATH='a.csv'
df=pd.read_csv(FILE_PATH)
source=np.array(df)
print(source)

NumDataPerClass = 10
X = source[:,[0,1,2,3]]
y = source[:,[4]]
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

def PercentCorrect(Inputs, targets, weights):
    N = len(targets)
    nCorrect = 0
    for n in range(N):
        OneInput = Inputs[n,:]
        if (targets[n] * np.dot(OneInput, weights) > 0):
            nCorrect +=1
    return 100*nCorrect/N


#-------------------------------------------------------------------
# Perceptron learning loop
#
# Random initialization of weights
#
w = np.random.randn(4)
print(w)
# What is the performance with the initial random weights?
#
print('Initial Percentage Correct: %6.2f' %(PercentCorrect(X_train, y_train, w)))
# Fixed number of iterations (think of better stopping criterion)
#
MaxIter=1000
# Learning rate (change this to see convergence changing)
#
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

#---------------------------------------------------------------
#ax.scatter(X_test[:,0], Y_test[:,1], c="b", s=4)
#ax.scatter(X[:,0], X[:,1], c="c", s=4)
#ax.set_xlim(-4, 8)
#ax.set_ylim(-4, 8)

plt.show()