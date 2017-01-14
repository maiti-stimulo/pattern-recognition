# -*- coding: utf-8 -*-
#Logistic regression
#importem les llibreries total omm parcialment segons ens interessa
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
import pylab


print 'Q2 get a new training and testing set that only contains the instances corresponding to shirt and jeans'
# indiquem que l'arxiu 3dclothing_labels_train.txt, està format per enters, la funcio sep espaia les dades.
# creem una variable per als index de train
training_data = np.load('3dclothing_train.npy')
training_labels = np.array([l.strip('\n\r').split(',') for l in open('3dclothing_labels_train.txt')])[:,-1]
# creem  una variable de training polo indexada per les eqtiquetes
# creem  una variable de training shirt indexada per les eqtiquetes
# where::Return elements, either from x or y, depending on condition
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
# nomes agafarem les etiquetes 'polo shirt' i ' shirt', les altres queden fora
training_polo_idx_labels = np.array(np.where(training_labels=='polo shirt'))[0]
print "training_polo_idx_labels"
print training_polo_idx_labels
training_shirt_idx_labels = np.array(np.where(training_labels=='shirt'))[0]
print "training_shirt_idx_labels"
print training_shirt_idx_labels
# ordenem les dades
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.sort.html
# sort::Return a sorted copy of an array
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html
#concatenate::Join a sequence of arrays along an existing axis.
training_idx_labels = np.sort(np.concatenate((training_polo_idx_labels, training_shirt_idx_labels)))

# aquestes dues linies poden esser comentades, no afecten el resultat final exercici
print "training_idx_labels, conte les dades de la training_polo_idx_labels i la training_shirt_idx_labels ajuntades i ordenades"
print training_idx_labels

training_data = training_data[training_idx_labels]
training_labels = training_labels[training_idx_labels]

#repetim el mateix procès per les dades de test
testing_data = np.load('3dclothing_test.npy')
testing_labels = np.array([l.strip('\n\r').split(',') for l in open('3dclothing_labels_test.txt')])[:,-1]

testing_polo_idx_labels = np.array(np.where(testing_labels=='polo shirt'))[0]
testing_shirt_idx_labels = np.array(np.where(testing_labels=='shirt'))[0]

testing_idx_labels = np.sort(np.concatenate((testing_polo_idx_labels, testing_shirt_idx_labels)))

testing_data = testing_data[testing_idx_labels]
testing_labels = testing_labels[testing_idx_labels]

#LogisticRegression::(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
#StratifiedKFold(y, n_folds=3, shuffle=False, random_state=None)
#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# elevem els valors de i a 10**-7 fins a 10**8,acumulatius
C_params = [10**i for i in range(-7,8)]
C_scores = []

for c in C_params:
    scores = []
    LogReg = LogisticRegression(C=c)
# Stratified K-Folds cross-validator
#Provides train/test indices to split data in train/test sets.
#This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class
#http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
# Hint:use log-scale for the C value in the plot. Hint: Since we do not have a lot of training data, use 15 folds to ensure the train set will be large enough
    skf = StratifiedKFold(training_labels, n_folds=15)
# ales dades de training ara les anomenen X
    X = training_data
# a les etiquetes de training les anomenem y
    y = training_labels
# segons la documentació StratifiedKFold ens retorna un conjunt de índex de training i testing. El bucle for ens permet recòrrer simultàniament els índex de training i els índex de testing
    for train_indices, test_indices in skf:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        LogReg.fit(X_train, y_train)
        score = LogReg.score(X_test, y_test)
        scores.append(score)
#max::Return the maximum of an array or maximum along an axis
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html
    max_score = np.max(scores)
#acumulem tots els valors de la matriu score
    C_scores.append((c,max_score))
# la X serà el primer valor de la matriu
X = [c[0] for c in C_scores]
# la y serà el segon valor de la matriu
Y = [c[1] for c in C_scores]

# defimin el plot títol, llegenda...
pylab.plot(X, Y, '-b')
pylab.xscale('log')
pylab.title('C vs. Accuracy')
pylab.legend(['Logistic regression'])
pylab.show()
# argmax::Returns the indices of the maximum values along an axis.
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html
max_accuracy = np.argmax(Y)
best_c = X[max_accuracy]
print 'El millor C es', best_c

LogReg = LogisticRegression(C=best_c)
# LogReg.fit: regressio kineal de train
LogReg.fit(training_data, training_labels)
# LogReg.score: regressio kineal de test
score = LogReg.score(testing_data, testing_labels)
print 'la precissió del model selecionat és', score


