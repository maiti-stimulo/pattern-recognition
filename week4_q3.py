# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
import pylab
# fem el mateix que en la Q2, però amb totes les dades
print 'Q3, Regression classifier with the complete training set'

training_data = np.load('3dclothing_train.npy')
training_labels = np.array([l.strip('\n\r').split(',') for l in open('3dclothing_labels_train.txt')])[:,-1]

testing_data = np.load('3dclothing_test.npy')
testing_labels = np.array([l.strip('\n\r').split(',') for l in open('3dclothing_labels_test.txt')])[:,-1]

#LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
#StratifiedKFold(y, n_folds=3, shuffle=False, random_state=None)

C_params = [10**i for i in range(-7,7)]
C_scores = []

for c in C_params:
    scores = []
    LogReg = LogisticRegression(C=c)
    skf = StratifiedKFold(training_labels, n_folds=15)
    X = training_data
    y = training_labels
    for train_indices, test_indices in skf:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        LogReg.fit(X_train, y_train)
        score = LogReg.score(X_test, y_test)
        scores.append(score)
    max_score = np.max(scores)
    C_scores.append((c,max_score))

X = [c[0] for c in C_scores]
Y = [c[1] for c in C_scores]

pylab.plot(X, Y, '-b')
pylab.xscale('log')
pylab.title('C vs. Accuracy')
pylab.legend(['Logistic regression'])
pylab.show()

max_accuracy = np.argmax(Y)
best_c = X[max_accuracy]
print 'Best C is', best_c

LogReg = LogisticRegression(C=best_c)
LogReg.fit(training_data, training_labels)
score = LogReg.score(testing_data, testing_labels)
print 'Accuracy for selected model is', score
