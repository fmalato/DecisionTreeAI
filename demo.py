import csv
import sys
import copy
import random

import DecisionTree as dt
import Heuristics as heur
import Utils as util
import CrossValidation as cv

# For great datasets, there could be a very high number of recursions while building the tree. I set it up to
# avoid sys errors.
sys.setrecursionlimit(1000000)
# Reads the .csv file and converts it in an usable dataset.
csvFile = csv.reader(file('/home/federico/Scrivania/Intelligenza Artificiale/Data Sets/playtennis.csv'), delimiter=",")
inputFile = list(csvFile)
# The first line of the .csv file is a list containing the attributes.
attributes = inputFile[0]
# That's why the target attribute should be in the last position in your .csv file. If you'd like to change,
# then change the following line as well.
targetAttr = attributes[len(attributes) - 1]
# Makes a copy of the examples, removing the attributes.
examples = copy.deepcopy(inputFile)
del examples[0]
# Just makes a copy of the 'examples' list, putting them in a dictionary.
trainingSet = []
for line in examples:
    trainingSet.append(dict(zip(attributes, [datum.strip() for datum in line])))
# Useful for some datasets, where the target attribute's values are already sorted.
random.shuffle(examples)
# Useful to make sure that there's a correct split of the dataset.
train, validation = cv.train_test_split(examples, len(examples)*0.5, len(examples) - 1)
# While coding, I found that some algorithm just performed better using a dictionary, some others were better
# using a list... so I've chosen to keep them both.
training = []
for line in train:
    training.append(dict(zip(attributes, [datum.strip() for datum in line])))
# Construction of the decision tree, based on a dictionary, a list of attributes and a target attribute. If you
# want to change target attribute, just make sure to modify it also in your data set.
decTree = dt.decisionTreeLearning(trainingSet, attributes, targetAttr)
# As long as I couldn't find any useful package, I provided also a fast function to print the tree. That was
# inspired by Christopher Roach in the "archive.oreilly.com" website. I'll also report that in the README file.
util.printTree(decTree, "")
# I provided ths due to the pluralityValue() function. I found out that makes it perform pretty well so... why not?
list = [x[attributes.index(targetAttr)] for x in examples]
""" After having the dataset structured in the right way, immediately calculates the plurality value, so
    it can be used in the following tests."""
default = dt.pluralityValue(list)
# That's a test I run to prove that the classification works.
classification = cv.classify(decTree, validation, attributes, default)
# Some indexes and some print to ensure everything's fine.
k = 0
correct = 0
for el in classification:
    actual = validation[classification.index(el)][attributes.index(targetAttr)]
    print "Input: " + str(validation[k])
    print "Atteso: " + str(validation[k][attributes.index(targetAttr)]) + "  " + "Trovato: " + str(el)
    if actual == el:
        correct += 1
    k += 1
# Some stats to look for problems and so on. At the moment, it performs really bad when the dataset's
# splitted by half. Maybe overfitting?
print "Numero di input testati: " + str(len(validation)) + "  " + " Numero di input corretti: " + str(correct)
total = float(len(validation))
ratio = ((float(correct))/(total))
print "Percentuale di correttezza: " + str((ratio)*100) + "%"

def kFoldCrossValidation(tree, k, examples, target, attributes):
    foldErrT = 0
    foldErrV = 0
    for fold in range(1, k):
        # Must change last two parameters in the function call...
        trainSet, validSet = cv.train_test_split(examples, len(examples)*0.5, len(examples) - 1)
        classification = cv.classify(tree, validSet, attributes, default)
        foldErrT += errorRate(classification, trainSet, target, attributes)
        foldErrV += errorRate(classification, validSet, target, attributes)
    return foldErrT / k, foldErrV / k

def errorRate(classification, vector, target, attributes):
    errT = 0
    errV = 0
    for element in vector:
        # classification has 6 elements, while trainSet has 8, using that kind of split. Must check it.
        if classification[vector.index(element)] != element[attributes.index(target)]:
            errT += 1
            errV += 0.5
    return errT, errV

foldErrT, foldErrV = kFoldCrossValidation(decTree, 5, examples, targetAttr, attributes)
print "FoldErrT: " + str(foldErrT) + "  FoldErrV: " + str(foldErrV)