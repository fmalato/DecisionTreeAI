import numpy as np
import csv
import sys
import copy
import random

import DecisionTree as dt
import Heuristics as heur
import Utils as util
import CrossValidation as cv

sys.setrecursionlimit(1000000)

csvFile = csv.reader(file('/home/federico/Scrivania/Intelligenza Artificiale/Data Sets/winequality-white.csv'), delimiter=";")
inputFile = list(csvFile)
attributes = inputFile[0]
targetAttr = attributes[len(attributes) - 1]

examples = copy.deepcopy(inputFile)
del examples[0]

trainingSet = []
for line in examples:
    trainingSet.append(dict(zip(attributes, [datum.strip() for datum in line])))

random.shuffle(examples)

train, validation = cv.train_test_split(examples, len(examples)*0.5, len(examples) - 1)

training = []
for line in train:
    training.append(dict(zip(attributes, [datum.strip() for datum in line])))

decTree = dt.decisionTreeLearning(trainingSet, attributes, targetAttr)
util.printTree(decTree, "")

list = [x[attributes.index(targetAttr)] for x in examples]
default = dt.pluralityValue(list)

#cv.classify(decTree, [attr for attr in attributes if attr != targetAttr], validation, lista)
classification = cv.classify(decTree, validation, attributes, default)
k = 0
correct = 0
for el in classification:
    actual = validation[classification.index(el)][attributes.index(targetAttr)]
    print "Input: " + str(validation[k])
    print "Atteso: " + str(actual) + "  " + "Trovato: " + str(el)
    if actual == el:
        correct += 1
    k += 1
print "Numero di input testati: " + str(len(validation)) + "  " + " Numero di input corretti: " + str(correct)
total = float(len(validation))
ratio = ((float(correct))/(total))
print "Percentuale di correttezza: " + str((ratio)*100) + "%"