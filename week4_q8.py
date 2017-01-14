# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import precision_recall_curve
import pylab

print 'Q8'
# carregem les dades de training robot_waiter_fries_scores.npy
# //////////Load and parse training data.Parse????????
# ////////per saber la longitut de les dades hem de fer un shape instead len???????
fries_scores = np.load('robot_waiter_fries_scores.npy')
print 'fries_scores.shape =', fries_scores.shape

# where::Return elements, either from x or y, depending on condition
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
# generem tres variables:
# una pels index que son true.
# una pels index que son false
# ///////// una per les dades true, necessitaria més explicació d'aquesta
# els index true han de ser majors o iguals a 0.8 /// exercici parla de 0.5??????????
# els index false han de ser menors  a 0.8 /// exercici parla de 0.5??????????
fries_scores_true_idx = np.array(np.where(fries_scores>=0.8))
fries_scores_false_idx = np.array(np.where(fries_scores<0.8))
fries_scores_true = fries_scores[fries_scores_true_idx]
# carregem les etiquetes robot_waiter_fries_labels.npy
# ////////per saber la longitut de les dades hem de fer un shape instead len???????
fries_labels = np.load('robot_waiter_fries_labels.npy')
print 'fries_labels.shape =', fries_labels.shape
# where::Return elements, either from x or y, depending on condition
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
# generem tres variables:
# una pels index de les etiquetes que son true.
# una pels index index de les etiquetes que son false
# ///////// una per les etiquetes true, necessitaria més explicació d'aquesta
fries_labels_true_idx = np.array(np.where(fries_labels==True))
fries_labels_false_idx = np.array(np.where(fries_labels==False))
fries_labels_true = fries_labels[fries_labels_true_idx]

print "Prediccions que son veritat:"
# //////////////com puc imprimir la len??????????????????
print fries_scores_true_idx
print 'Etiquetes que son veritat:'
print fries_labels_true_idx

# en el següent bloc defimin:
# etiquetes i prediccions que son veritat
# prediccions que son veritat amb etiquetes falses
# prediccions falses amb etiquetes que son veritat
# etiquetes i orediccions falses
#////////////////////////////////////////////////////////////////////
# TP = true predictions and true labels
TP_set = np.intersect1d(fries_scores_true_idx, fries_labels_true_idx)
TP = TP_set.shape[0]

# FP = true predictions and false labels
#FP_set = np.setdiff1d(fries_scores_true_idx, fries_labels_true_idx)
FP_set = np.intersect1d(fries_scores_true_idx, fries_labels_false_idx)
FP = FP_set.shape[0]

# FN = false predictions and true labels
#FN_set = np.setdiff1d(fries_labels_true_idx, fries_scores_true_idx)
FN_set = np.intersect1d(fries_scores_false_idx, fries_labels_true_idx)
FN = FN_set.shape[0]

# TN = false predictions and false labels
TN_set = np.intersect1d(fries_scores_false_idx, fries_labels_false_idx)
TN = TN_set.shape[0]
#///////////////////////////////////////////////////////77
# imprimin els quatre resultats
#////////////// imprimir la len de cadascun d'ells?
print 'TP_set (true predictions and true labels):'
print TP_set

print 'FP_set(true predictions and false labels):'
print FP_set

print 'FN_set(false predictions and true labels):'
print FN_set

print 'TN_set (false predictions and false labels):'
print TN_set

# en aquest bloc definim els ratis i el grau de certessa
#///////////////////////////////////////////////////////////
# percentatge de certessa
# percentatge d'error
clf_accuracy = float(TP+TN)/float(TP+FP+FN+TN)
error_rate = float(FP+FN)/float(TP+FP+FN+TN)
# veritables positius i negatius
true_pos_rate = float(TP)/float(TP+FP)
true_neg_rate = float(TN)/float(TN+FN)
# falsos positius i negatius
false_pos_rate = float(FP)/float(FP+TN)
false_neg_rate = float(FN)/float(FN+TP)

balanced_error_rate = 0.5 * (false_pos_rate+false_neg_rate)
F1_score = 2.0*TP/float(2.0*TP+FP+FN)
beta = 0.5
Fbeta_score = (1+beta**2)*TP/((1+beta**2)*TP + (beta**2*FN)+ FP)

print 'Percentatge de certessa:', clf_accuracy
print 'Percentatge de error:', error_rate
print 'Percentatges de positius certss:', true_pos_rate
print 'Percentatge de negatius certs:', true_neg_rate
print 'Percentatge de falsos positius:', false_pos_rate
print 'Percentatge de falsos negatius:', false_neg_rate
print 'Balanced error rate:', balanced_error_rate
print 'F1-score:', F1_score
print 'F-beta score (beta=%.2f): %f' % (beta,Fbeta_score)
#///////////////////////////////////////////////////////////////////////

# definim dues noves variables
#y_trueper les etiquetes
#y_scores per dades de test
y_true = fries_labels
y_scores = fries_scores

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
# defimin el plot
pylab.plot(recall, precision, '-b')
pylab.title('precision-recall curve')
pylab.legend(['French fries classification'])
pylab.show()
