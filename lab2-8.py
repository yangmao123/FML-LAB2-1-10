# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 23:26:55 2020

@author: 75965
"""
import matplotlib.pyplot as plt
import numpy as np
# Scikitlearn can do it for us
#
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

NumDataPerClass = 400
# Two-class problem, distinct means, equal covariance matrices
#
m1 = [[0, 5]]
m2 = [[5, 0]]
C = [[2, 1], [1, 2]]
# Set up the data by generating isotropic Guassians and
# rotating them accordingly
#
A = np.linalg.cholesky(C)
U1 = np.random.randn(NumDataPerClass,2)
X1 = U1 @ A.T + m1
U2 = np.random.randn(NumDataPerClass,2)
X2 = U2 @ A.T + m2

X = np.concatenate((X1, X2), axis=0)

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(X1[:,0], X1[:,1], c="r", s=4)
ax.scatter(X2[:,0], X2[:,1], c="b", s=4)
ax.scatter(X[:,0], X[:,1], c="c", s=4)
ax.set_xlim(-4, 8)
ax.set_ylim(-4, 8)

plt.show()

labelPos = np.ones(NumDataPerClass)
labelNeg = -1.0 * np.ones(NumDataPerClass)
y = np.concatenate((labelPos, labelNeg))

#----------------------------------------------------------------
rIndex = np.random.permutation(2*NumDataPerClass)
Xr = X[rIndex]
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

# Main Loop
#
model = Perceptron()
model.fit(X_train, y_train)
yh_train = model.predict(X_train)
print("Accuracy on training set: %6.2f" %(accuracy_score(yh_train, y_train)))
yh_test = model.predict(X_test)
print("Accuracy on test set: %6.2f" %(accuracy_score(yh_test, y_test)))
if (accuracy_score(yh_test, y_test) > 0.99):
  print("Wow, Perfect Classification on Separable dataset!")

    

#---------------------------------------------------------------
#ax.scatter(X_test[:,0], Y_test[:,1], c="b", s=4)
#ax.scatter(X[:,0], X[:,1], c="c", s=4)
#ax.set_xlim(-4, 8)
#ax.set_ylim(-4, 8)

plt.show()




