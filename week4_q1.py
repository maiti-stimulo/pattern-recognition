# -*- coding: utf-8 -*-
import numpy as np
# cridem un determinada biblioteca i importem la part que ens interessa.
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
import pylab

print 'Q1 Apartat 1'
# indiquem que l'arxiu iris_idx_train.txt, està format per enters, la funcio sep espaia les dades.
# creem una variable per als index de train
# les dades quedaran representades de la següent manera [55   3  85  18 149  20 ...	
idx_train = np.fromfile('iris_idx_train.txt', int, sep=" ")

#creem una variable per index de test
idx_test = np.fromfile('iris_idx_test.txt', int, sep=" ")

# \r (Carriage Return) - moves the cursor to the beginning of the line without advancing to the next line
# \n (Line Feed) - moves the cursor down to the next line without returning to the beginning of the line
# el tipus de dades que conté iris.data es :5.1,3.5,1.4,0.2,Iris-setosa
# del conjunt de dades eliminem la última columna
# eliminem les comes que separen les dades per poder generar la matriu.
# indiquem que les dades son float, ja que son numeros decimals
# generem una matriu amb totes les dades, però eliminant els elements que no necessitem
# el tipus de dades que ens genera és d'aquest tipus:'5.1' '3.5' '1.4' '0.2' 'Iris-setosa'
loaded_data = np.array([l.strip('\n\r').split(',') for l in open('iris.data')])
# plotejem com queda la matriu
print "loaded_data"
print loaded_data
# generem una matriu sense la última columna
loaded_data_without_labels = np.array([map(float, l.split(',')[:-1]) for l in open('iris.data')])
# hem plotejat les dades per veure si ho hem fet correctament, ara està comentat.
#print "loaded_data_without_labels"
#print loaded_data_without_labels
#estem generan una matriu que l'element que contè és una matriu
# Ara ja tenim les dades tal i com les volem i ja podem començar a treballar amb elles
# les dades de train seran les del index train
# les dades de train sense etiquetes seran les del index train without labels
# les dades de train labels seran les del index train data, però quedant-nos amb la última columna
train_data = loaded_data[idx_train]
train_data_without_labels = loaded_data_without_labels[idx_train]
train_labels = train_data[:,-1]
# aqui fem el mateix però amb les dades de test
test_data = loaded_data[idx_test]
test_data_without_labels = loaded_data_without_labels[idx_test]
test_labels = test_data[:,-1]
# defimin una funció knn 
def kNN(dataset, sample_idx, labels, k):
    sample = dataset[sample_idx]
    dist = cdist(np.atleast_2d(sample), dataset)
    #cdist és una funció que retorna la distància entre dos punts donats:
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    #atleast::View inputs as arrays with at least two dimensions.
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_2d.html
    # ens retorna An array, or tuple of arrays, each with a.ndim >= 2. Copies are avoided where possible, and views with two or more dimensions are returned.
    #En resum, necessitem que dataset i sample tinguin el mateix número de columnes (és a dir, número de coordenades) per a que cdist funcioni, i atleast_2d ens fa això.

    min_k = np.argsort(dist[0])[1:k+1]
    #argosort:: returns the indices that would sort an array.
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
    # ens retorna Array of indices that sort a along the specified axis. If a is one-dimensional, a[index_array] yields a sorted a.
    min_labels = labels[min_k]
    
    return mode(min_labels)[0][0] == labels[sample_idx]


print 'Q1 apartat 2:'

print '\nLegend = (k, correctly classified samples, accuracy)'

print '\nIris: classification for training'



# creem una funcio que ens generi una matriu que ens classificara els elements de training
training_classification = []
#range :: it generates a list of numbers, which is generally used to iterate over with for loops
#el range va de 1 fins a 10, però utilitzarem els elements parells
#discriminem els valors parells i imparells dividint-los entre dos i mirant el quocient
for k in [n for n in range(1,10) if n % 2 != 0]:
     correct_samples = 0
    # a la línia 81 comencem un bucle for amb el que recorrem tota la sèrie de valors k que volem posar a prova. Cada cop que entrem al bucle posem el número de samples correctes a zero, calculem el que necessitem (línies 85 i 87) i guardem el número de samples correctes (línia 93)
#els valors que utilitzarem seran els que tinguin quocient !=0
     for i in range(0,len(train_data_without_labels)):
# en aquest cas el range va de 0 fins a la longitut del train_data_wihout_labels
        if kNN(train_data_without_labels, i, train_labels, k):
            correct_samples = correct_samples+1
#append = anexar.
#Append values to the end of an array
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.append.html
#anexem les dades correctes a la matriu del training classsification
     training_classification.append((k, correct_samples, 1.0-float((len(train_data_without_labels)-correct_samples))/len(train_data_without_labels)))
# ens dona un rsultat d'aquest tipus (1, 45, 0.9)
# k es 1, correct samples es 45 ,1.0-float((len(train_data_without_labels)-correct_samples))/len(train_data_without_labels)))" és 0.9

best_training_classification = (0,0,0)
# ens retornarà tres valors separts per comes, indicant el millor valor
for item in training_classification:
    if item[2] > best_training_classification[2]:
# la dada que millor funciona serà la que ocupa la posició 2, tinguen en compte que sempre comencem per la zero.
        best_training_classification = item
    print item

print 'Best is', best_training_classification


# la classification for testing and training funcionen exactament igual que la classification per training, ja no les comento

print '\nIris: classification for testing'
testing_classification = []

for k in [n for n in range(1,10) if n % 2 != 0]:
    correct_samples = 0
    for i in range(0,len(test_data_without_labels)):
        if kNN(test_data_without_labels, i, test_labels, k):
            correct_samples = correct_samples+1
    testing_classification.append((k, correct_samples, 1.0-float((len(test_data_without_labels)-correct_samples))/len(test_data_without_labels)))

best_testing_classification = (0,0,0)
for item in testing_classification:
    if item[2] > best_testing_classification[2]:
        best_testing_classification = item
    print item
print 'Best is', best_testing_classification

print" Q1, 3DClothing"

clothing_training_data = np.load('3dclothing_train.npy')
# indiquem que l'arxiu 3dclothing_labels_train.txt, està format per enters, la funcio sep espaia les dades.
# creem una variable per als index de train
clothing_training_labels = np.array([l.strip('\n\r').split(',') for l in open('3dclothing_labels_train.txt')])[:,-1]
#discriminem els valors parells i imparells dividint-los entre dos i mirant el quocient
x = [n for n in range(1,10) if n % 2 != 0]

print '\nClothing: classification for training'
# definim una matriu de classificadors de training
training_classification = []
# definim una matriu y que guarda tots els valors obtinguts de y.append
y = []
for k in x:
# els valos k que siguin imparells
    correct_samples = 0
    for i in range(0,len(clothing_training_data)):
#definim un range que va de 0 a la long de la matriu clothing_training_data
#////////////////////////////////////////////////////////////////////////
        if kNN(clothing_training_data, i, clothing_training_labels, k):
            correct_samples = correct_samples+1
# els correct samples s'incementaran de 1 en 1 fins a arribar a la len de la matriu
#no entenc pq repetim dues vegades mla mateixa funció ??????????????????
    y.append(1.0-float((len(clothing_training_data)-correct_samples))/len(clothing_training_data))
    training_classification.append((k, correct_samples, 1.0-float((len(clothing_training_data)-correct_samples))/len(clothing_training_data)))
#///////////////////////////////////////////////////////////////////////////
best_training_classification = (0,0,0)
for item in training_classification:
    if item[2] > best_training_classification[2]:
        best_training_classification = item
    print item

print 'Best is', best_training_classification

#repetim el mateix procès que en les etiquetes de train, però ara amb les de test.
clothing_testing_data = np.load('3dclothing_test.npy')
clothing_testing_labels = np.array([l.strip('\n\r').split(',') for l in open('3dclothing_labels_test.txt')])[:,-1]

print '\nClothing: classification for testing'
testing_classification = []
y = []

for k in x:
    correct_samples = 0
    for i in range(0,len(clothing_testing_data)):
        if kNN(clothing_testing_data, i, clothing_testing_labels, k):
            correct_samples = correct_samples+1
    y.append(1.0-float((len(clothing_testing_data)-correct_samples))/len(clothing_testing_data))
    testing_classification.append((k, correct_samples, 1.0-float((len(clothing_testing_data)-correct_samples))/len(clothing_testing_data)))

best_testing_classification = (0,0,0)
for item in testing_classification:
    if item[2] > best_testing_classification[2]:
        best_testing_classification = item
    print item
print 'Best is', best_testing_classification


pylab.plot(x, y, '-b')
# plotejem les x les y amb un guio blau
pylab.title('k vs accuracy')
# títol de la gràfica
pylab.legend(['accuracy'])
# llegenda de la fràfica plotejada
pylab.show()
