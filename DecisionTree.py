import numpy as np
import graphviz as gv
import math
import anytree
from anytree.exporter import dotexporter as dotexp
import matplotlib.pyplot as plt
import copy
import networkx as netx
import csv

csvFile = csv.reader(file('/home/federico/Scrivania/Intelligenza Artificiale/Data Sets/playtennis.csv'), delimiter=",")
trainingSet = list(csvFile)
attributes = trainingSet[0]
attrIndex = range(len(attributes))
del trainingSet[0]
A = np.zeros(len(attributes))

dictionarySet = {}
for j in range(len(attributes)):
    dictionarySet[attributes[j]] = [i[j] for i in trainingSet]

"""targets = copy.deepcopy(dictionarySet['PlayTennis'])
del dictionarySet['PlayTennis']
del attributes[attrIndex[4]]
del attrIndex[4]"""
#targets = range(len(attributes))

def decisionTreeLearning(trainingSet, attributes, attrIndex, dictionarySet, targetAttr):

    localTraining = copy.deepcopy(trainingSet)
    localAttr = copy.deepcopy(attributes)
    localIndex = copy.deepcopy(attrIndex)
    localDict = copy.deepcopy(dictionarySet)

    vals = [record for record in dictionarySet[targetAttr]]

    if not dictionarySet or (len(localAttr) - 1) <= 0:
        return pluralityValue(localTraining)
    elif localDict[targetAttr].count(localDict[targetAttr][0]) == len(localDict[targetAttr]):
        return localDict[targetAttr][0]
    else:
        maxGain, maxGainIndex = chooseBestAttribute(localAttr, localDict, localIndex, targetAttr)

        A = localDict[localAttr[maxGainIndex]]
        A = set(A)

        best = localAttr[maxGainIndex]
        subDictionary = arraySubtraction(localDict, localAttr[maxGainIndex])
        subAttributes = arraySubtraction(localAttr, maxGainIndex)
        subAttrIndex = arraySubtraction(localIndex, maxGainIndex)
        for v in localTraining:
            del v[maxGainIndex]
        decTree = {best: {}}
        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for val in A:
            # Create a subtree for the current value under the "best" field
            subtree = decisionTreeLearning(localTraining, subAttributes, subAttrIndex, subDictionary, targetAttr)

            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            decTree[best][val] = subtree

    return decTree

def getExamples(data, best, value):
    A = []
    for record in data[best]:
        if record == value:
            A.append(record)
    return A

def arraySubtraction(array, attribute):
    del array[attribute]
    return array

def pluralityValue(set):
    lst = set[:]
    highestFreq = 0
    mostFreq = None

    for val in unique(lst):
        if lst.count(val) > highestFreq:
            mostFreq = val
            highestFreq = lst.count(val)

    return mostFreq

def chooseBestAttribute(attr, dict, index, target):
    k = 0
    maxGain = 0
    maxGainIndex = 0
    gains = np.zeros(len(attr))
    for j in attr:
        gains[k] = gain(dict[j], index[k], k)
        if gains[k] > maxGain:
            maxGain = gains[k]
            maxGainIndex = k
        k += 1
    return maxGain, maxGainIndex

def unique(lst):
    """
    Returns a list made up of the unique values found in lst.  i.e., it
    removes the redundant values in lst.
    """
    lst = lst[:]
    uniqueLst = []

    # Cycle through the list and add each value to the unique list only once.
    for item in lst:
        if uniqueLst.count(item) <= 0:
            uniqueLst.append(item)

    # Return the list with all redundant values removed.
    return uniqueLst


def entropy(data, target_attr):
    """
    Calculates the entropy of the given data set for the target attribute.
    """
    val_freq = {}
    data_entropy = 0.0

    # Calculate the frequency of each of the values in the target attr
    for record in data:
        if (record in val_freq):
            val_freq[record] += 1.0
        else:
            val_freq[record] = 1.0

    # Calculate the entropy of the data for the target attribute
    for freq in val_freq.values():
        data_entropy += (-freq / len(data)) * math.log(freq / len(data), 2)

    return data_entropy


def gain(data, attr, target_attr):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """
    val_freq = {}
    subset_entropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for record in data:
        if(record in val_freq):
            val_freq[record] += 1.0
        else:
            val_freq[record] = 1.0

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        data_subset = [record for record in data if record == val]
        subset_entropy += val_prob * entropy(data_subset, target_attr)

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    return (entropy(data, target_attr) - subset_entropy)

dt = decisionTreeLearning(trainingSet, attributes, attrIndex, dictionarySet, 'PlayTennis')
#print anytree.RenderTree(dt)
#dotexp.DotExporter(dt).to_picture("dt.png")
print dt