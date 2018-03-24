import numpy as np
import graphviz as gv
import math
import anytree
from anytree.exporter import dotexporter as dotexp
import matplotlib.pyplot as plt
import copy
import networkx as netx
import csv

csvFile = csv.reader(file('/home/federico/Scrivania/Intelligenza Artificiale/Data Sets/shuttle.csv'), delimiter=",")
trainingSet = list(csvFile)
attributes = trainingSet[0]
attrIndex = range(len(attributes))
del trainingSet[0]
A = np.zeros(len(attributes))

dictionarySet = {}
for j in range(len(attributes)):
    dictionarySet[attributes[j]] = [i[j] for i in trainingSet]

targets = copy.deepcopy(dictionarySet['Target'])
del dictionarySet['Target']
del attributes[attrIndex[6]]
del attrIndex[6]

def decisionTreeLearning(trainingSet, attributes, attrIndex, dictionarySet):

    localTraining = copy.deepcopy(trainingSet)
    localAttr = copy.deepcopy(attributes)
    localIndex = copy.deepcopy(attrIndex)
    localDict = copy.deepcopy(dictionarySet)
    #values = [record["density"] for record in trainingSet]
    for i in range(len(trainingSet)):
        if (targets[0] != targets[i+1]):
            break
        elif targets[0] == targets[i+1] and i == len(localTraining):
            return targets[0]
    if not localAttr:
        return pluralityValue(localTraining)
    else:
        k = 0
        maxGain = 0
        gains = np.zeros(len(localAttr))
        for j in localAttr:
            gains[k] = gain(localDict[j], localIndex[k], k)
            if gains[k] > maxGain:
                maxGain = gains[k]
                maxGainIndex = k
            k += 1
        A = localDict[localAttr[maxGainIndex]]
        A = set(A)
        #decTree = anytree.Node(localAttr[maxGainIndex])
        decTree = netx.Graph()
        decTree.add_node(localAttr[maxGainIndex], root=True)
        subDictionary = arraySubtraction(localDict, localAttr[maxGainIndex])
        subAttributes = arraySubtraction(localAttr, maxGainIndex)
        subAttrIndex = arraySubtraction(localIndex, maxGainIndex)
        for v in localTraining:
            del v[maxGainIndex]
        subTree = []
        for value in A:
            #exs = [e for e in dictionarySet[attributes[maxGainIndex]] if e == value]
            #exs = [filter(lambda e: e == v, A)] # Doesn't do what it should
            #newNode = anytree.Node(decisionTreeLearning(localTraining, subAttributes, subAttrIndex, subDictionary), parent=decTree)
            newNode = decisionTreeLearning(localTraining, subAttributes, subAttrIndex, subDictionary)
            decTree.add_path(newNode)
            #subTree.append(newNode)
            #subTree.name = attributes[maxGainIndex]
            #decTree.children = subTree
    return decTree

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

dt = decisionTreeLearning(trainingSet, attributes, attrIndex, dictionarySet)
#print anytree.RenderTree(dt)
#dotexp.DotExporter(dt).to_picture("dt.png")
netx.draw_networkx(dt)
plt.show()