# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
import pylab

print 'Q4'

training_data = np.load('3dclothing_train.npy')
training_labels = np.array([l.strip('\n\r').split(',') for l in open('3dclothing_labels_train.txt')])[:,-1]

training_polo_idx_labels = np.array(np.where(training_labels=='polo shirt'))[0]
training_shirt_idx_labels = np.array(np.where(training_labels=='shirt'))[0]

training_idx_labels = np.sort(np.concatenate((training_polo_idx_labels, training_shirt_idx_labels)))
#plotejem les etiquetes per veure si ho hem fet correctament, ara esta comentat
#print "training_idx_labels"
#print training_idx_labels

training_data = training_data[training_idx_labels]
#plotejem les dades per veure si ho hem fet correctament, ara esta comentat
#print "training_data"
#print training_data
training_labels = training_labels[training_idx_labels]

testing_data = np.load('3dclothing_test.npy')
testing_labels = np.array([l.strip('\n\r').split(',') for l in open('3dclothing_labels_test.txt')])[:,-1]

testing_polo_idx_labels = np.array(np.where(testing_labels=='polo shirt'))[0]
testing_shirt_idx_labels = np.array(np.where(testing_labels=='shirt'))[0]

testing_idx_labels = np.sort(np.concatenate((testing_polo_idx_labels, testing_shirt_idx_labels)))
#plotejem les dades per veure si ho hem fet correctament, ara esta comentat
#print "testing_idx_labels"
#print testing_idx_labels

testing_data = testing_data[testing_idx_labels]
testing_labels = testing_labels[testing_idx_labels]
#plotejem les dades per veure si ho hem fet correctament, ara esta comentat
#print "testing_data"
#print testing_data
#print "testing_labels"

#aquest codi és el mateix que el del Q2
#///////////////////////////////////////////////
# Hem fet un altre cop el mateix que al Q2 però només per a poder analitzar les dades que ens demana l'exercici. Com que ja ho teníem es "copiar i enganxar".
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
# defimin el plot
pylab.plot(X, Y, '-b')
pylab.xscale('log')
pylab.title('C vs. Accuracy')
pylab.legend(['Regressio lineal'])
pylab.show()

max_accuracy = np.argmax(Y)
best_c = X[max_accuracy]
print 'la millor c es', best_c

LogReg = LogisticRegression(C=best_c)
P = dir(LogReg)
LogReg.fit(training_data, training_labels)
score = LogReg.score(testing_data, testing_labels)
print 'la precissió del model selecionat és', score
Q = dir(LogReg)
#////////////////////////////////////////////////

print 'Nous atributs del nostre objecte de regressió logística després de entrenament:\n',[item for item in Q if item not in P],'\n'
print 'Segons documentació reutilitzarem les dades que obtenim de coef_ and intercept_,'
print 'que son els termes que ens requereix la regressió logística.\n'
print 'Veure https://en.wikipedia.org/wiki/Logistic_regression#Interpretation_of_these_terms'
print '\n'
print 'així mateix, els scikit tutorials recomamen utilitzar els pickles de python,'
print 'amb el qual està bast el seu model de persistència,'
print 'donant més eficiencia al model.'


print "BIAS:"
print LogReg.intercept_

print "THETA:"
print LogReg.coef_


import pickle
s = pickle.dumps(LogReg)
LogReg2 = pickle.loads(s)
score2 = LogReg2.score(testing_data, testing_labels)
print 'Testejant pickle...'
#print 'Accuracy for saved-then-loaded-again model is', score2int testing_labels
print 'Precisió de les dades guardades i carregades altre cop al model', score2

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
print 'La millor C es', best_c

LogReg = LogisticRegression(C=best_c)
P = dir(LogReg)
LogReg.fit(training_data, training_labels)
score = LogReg.score(testing_data, testing_labels)
print 'Precissió model seleccionat', score
Q = dir(LogReg)

print 'Nous atributs del nostre objecte de regressió logística després de entrenament:\n',[item for item in Q if item not in P],'\n'
print 'Segons documentació reutilitzarem les dades que obtenim de coef_ and intercept_,'
print 'que son els termes que ens requereix la regressió logística.\n'
print 'Veure https://en.wikipedia.org/wiki/Logistic_regression#Interpretation_of_these_terms'
print '\n'
print 'així mateix, els scikit tutorials recomamen utilitzar els pickles de python,'
print 'amb el qual està bast el seu model de persistència,'
print 'donant més eficiencia al model.'



print "BIAS:"
print LogReg.intercept_

print "THETA:"
print LogReg.coef_


import pickle
s = pickle.dumps(LogReg)
LogReg2 = pickle.loads(s)
score2 = LogReg2.score(testing_data, testing_labels)
print 'Testejant	 pickle...'
print 'Precisió de les dades guardades i carregades altre cop al model', score2

