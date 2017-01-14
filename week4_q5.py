# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

print 'Q5'
# carregem les dades de l'arxiu jain.txt
# arxiu d'aquest tipus 0.85	17.45	2
loaded_data = np.array([map(float, l.split()) for l in open('jain.txt')])
print len(loaded_data)
#len és373
#random:: aleatories
#https://docs.scipy.org/doc/numpy/reference/routines.random.html
np.random.shuffle(loaded_data)
# ara les dades son les mateixes del jain.txt, però desordenades
training_data = loaded_data[:len(loaded_data)/2,:]
print len(training_data)
#len es 186
testing_data = loaded_data[len(loaded_data)/2:len(loaded_data),:]
print len(testing_data)
#len es 187

#print "training_data"
#print training_data
#print "testing_data"
#print testing_data

# de les dades de training
# traiem la última columna de la matriu, que son les etiquetes
training_samples = training_data[:,:-1]
# ens quedem amb les etiquetes
training_labels = training_data[:,-1]

#print "training_samples"
#print training_samples
#print "training_labels"
#print training_labels

# de les dades de testing
# traiem la última columna de la matriu, que son les etiquetes
testing_samples = testing_data[:,:-1]
# ens quedem amb les etiquetes
testing_labels = testing_data[:,-1]

#print "testing_samples"
#print testing_samples
#print "testing_labels"
#print testing_labels

# entremen la linear SV machine
my_linearsvc = LinearSVC()
my_linearsvc.fit(training_samples, training_labels)
LinearSVC_score = my_linearsvc.score(testing_samples, testing_labels)
print 'LinearSVC() score:', LinearSVC_score

# entrenem la RBF SV machine
my_svc = SVC()
my_svc.fit(training_samples, training_labels)
SVC_score = my_svc.score(testing_samples, testing_labels)
print 'RBF score:', SVC_score

print training_samples.shape
print my_svc.support_vectors_.shape

# Adapted from: Gael Varoquaux, Andreas Muller; "Classifier comparison"
# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
def paint_decision_functions(data, labels, clf):
    from matplotlib.colors import ListedColormap
    import pylab
    cm = pylab.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    x_min, x_max = data[:, 0].min() - .5, data[:, 0].max() + .5
    y_min, y_max = data[:, 1].min() - .5, data[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    pylab.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    pylab.scatter(data[:, 0], data[:, 1], c=labels, cmap=cm_bright)
    pylab.xlim(xx.min(), xx.max())
    pylab.ylim(yy.min(), yy.max())
    pylab.xticks(())
    pylab.yticks(())
    pylab.show()

paint_decision_functions(testing_samples, testing_labels, my_linearsvc)
paint_decision_functions(testing_samples, testing_labels, my_svc)
print " un cop analitzats els dos plots la RBF classifica d'una manera molt més acurada que la SVC."
print " RBF genera dos grans grups de dades, i RBF agrupales dades en diferents grups, poc difereniadors"
#print 'Linear SVC can\'t find a separating hyperplane that classifies correcly, many samples'
#print 'end falling under the same hyperplanes. However, RBF is able to classify almost all of'
#print 'the samples properly.'
