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
csvFile = csv.reader(file('/home/federico/Scrivania/Intelligenza Artificiale/Data Sets/nurseryClassifier.csv'), delimiter=",")
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
# I provided ths due to the pluralityValue() function. I found out that makes it perform pretty well so... why not?
list = [x[attributes.index(targetAttr)] for x in examples]
# After having the dataset structured in the right way, immediately calculates the plurality value, so
# it can be used in the following tests.
default = dt.pluralityValue(list)

def unknownDataTest(examples, attributes, target, times):
    # The number of examples becomes bigger and bigger, so the tree becomes more accurate after every
    # iteration.
    list = [x[attributes.index(targetAttr)] for x in examples]
    default = dt.pluralityValue(list)
    giniScores, entrScores, misclassScores = cv.kFoldCrossValidation(times, examples, target, attributes, default)
    print
    print "Media giniScores: " + str(giniScores) + "  Media entrScores: " + str(entrScores) +\
          "  Media misClassScores: " + str(misclassScores)

#dec = dt.decisionTreeLearning(trainingSet, attributes, targetAttr, criterion='missclass')
#util.printTree(dec, "")
unknownDataTest(examples, attributes, targetAttr, 5)
