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
# random.shuffle(examples)
# Useful to make sure that there's a correct split of the dataset.
"""train, validation = cv.train_test_split(examples, len(examples)*0.5, len(examples) - 1)
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
util.printTree(decTree, "") """
# I provided ths due to the pluralityValue() function. I found out that makes it perform pretty well so... why not?
list = [x[attributes.index(targetAttr)] for x in examples]
# After having the dataset structured in the right way, immediately calculates the plurality value, so
# it can be used in the following tests.
default = dt.pluralityValue(list)
# That's a test I run to prove that the classification works.
"""classification = cv.classify(decTree, validation, attributes, default)
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
# splitted by half. Maybe overfitting? Nope, pruning already implemented.
print "Numero di input testati: " + str(len(validation)) + "  " + " Numero di input corretti: " + str(correct)
total = float(len(validation))
ratio = ((float(correct))/(total))
print "Percentuale di correttezza: " + str((ratio)*100) + "%"

foldErrT, foldErrV = cv.kFoldCrossValidation(decTree, 5, examples, targetAttr, attributes, default)
print "FoldErrT: " + str(foldErrT) + "  FoldErrV: " + str(foldErrV)"""

def unknownDataTest(examples, attributes, target, times):
    # The number of examples becomes bigger and bigger, so the tree becomes more accurate after every
    # iteration.
    for j in range(times):
        print "####################"
        print "### iterazione " + str(j+1) + " ###"
        print "####################"
        # random.shuffle(examples) ## This line just shuffles the examples array, so tests are randomized.
        train, validation = cv.train_test_split(examples, (len(examples)/times) * j, len(examples) - 1)
        training = []
        for line in train:
            training.append(dict(zip(attributes, [datum.strip() for datum in line])))
        decisionTree = dt.decisionTreeLearning(training, attributes, target)
        util.printTree(decisionTree, "")
        classification = cv.classify(decisionTree, validation, attributes, default)
        # Starting the test part
        k = 0
        correct = 0
        for el in classification:
            actual = validation[k][attributes.index(targetAttr)]
            """print "Input: " + str(validation[k])
            print "Atteso: " + str(validation[k][attributes.index(targetAttr)]) + "  " + "Trovato: " + str(el) """
            if actual == el:
                correct += 1
            k += 1
        print "Numero di input testati: " + str(len(validation)) + "  " + " Numero di input corretti: " + str(correct)
        total = float(len(validation))
        ratio = ((float(correct)) / (total))
        print "Percentuale di correttezza: " + str((ratio) * 100) + "%"
        foldErrT, foldErrV = cv.kFoldCrossValidation(decisionTree, 5, examples, targetAttr, attributes, default)
        print "FoldErrT: " + str(foldErrT) + "  FoldErrV: " + str(foldErrV)

# unknownDataTest(examples, attributes, targetAttr, 5)

def chooseBestAttribute(data, attributes, target, criterion, exs):
    data = data[:]
    bestGain = 0.0
    bestAttr = None

    if criterion != 'gini':
        for attr in attributes:
            if criterion == 'gain':
                gainAttr = heur.gain(data, attr, target)
                if (gainAttr >= bestGain and attr != target):
                    bestGain = gainAttr
                    bestAttr = attr
            else:
                gainAttr = heur.entropy(data, target)
                if (gainAttr >= bestGain and attr != target):
                    bestGain = gainAttr
                    bestAttr = attr
    else:
        bestGain, index = heur.giniIndex(exs, attributes)
        bestAttr = attributes[index]
    return bestAttr

def decisionTreeLearning(trainingSet, attributes, targetAttr, exs, samples, criterion='gain'):

    localTraining = copy.deepcopy(trainingSet)
    localAttr = copy.deepcopy(attributes)
    samples = copy.deepcopy(samples)
    values = [record[targetAttr] for record in trainingSet]

    default = dt.pluralityValue(localTraining)

    if not localTraining or ((len(localAttr) - 1) <= 0):
        return default
    elif values.count(values[0]) == len(values):
        return values[0]
    else:
        maxGainAttr = chooseBestAttribute(localTraining, attributes, targetAttr, criterion, samples)
        decTree = {maxGainAttr: {}}
        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for x in samples:
            del x[attributes.index(maxGainAttr)]
        for value in dt.getValues(trainingSet, maxGainAttr):
            # Create a subtree for the current value under the "best" field
            subtree = decisionTreeLearning(dt.getExamples(localTraining, maxGainAttr, value),
                                           [attr for attr in attributes if attr is not maxGainAttr],
                                           targetAttr, exs, samples, criterion)

            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            decTree[maxGainAttr][value] = subtree
    return decTree

dec = decisionTreeLearning(trainingSet, attributes, targetAttr,  examples, examples, criterion='gain')
print dec
