import pandas as pd
import numpy as np
import graphviz as gv
import math
import sklearn
from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import cross_val_score
import csv

csvFile = csv.reader(file('/home/federico/Scrivania/Intelligenza Artificiale/Data Sets/winequality-white.csv'), delimiter=";")
trainingSet = list(csvFile)
attributes = trainingSet[0]
del trainingSet[0]

dictionarySet = {}
for j in range(len(attributes)):
    dictionarySet[attributes[j]] = [i[j] for i in trainingSet]


def decisionTreeLearning(trainingSet, attributes):

    values = [record["density"] for record in trainingSet]

    if all(x["quality"] == x["quality"] for x in trainingSet):
        return trainingSet[0, "quality"]
    elif not attributes:
        return pluralityValue(trainingSet)
    else:
        for j in attributes:
            A = [np.argmax(dictionarySet[j])]
        decTree = tree.DecisionTreeClassifier(trainingSet, attributes)
        for v in A:
            ### exs = [e.A for e in trainingSet if e.A = v]
            exs = filter(lambda e: e.A ==v, trainingSet)
            subTree = decisionTreeLearning(exs, attributes - A)
            scores = cross_val_score(decTree, trainingSet, exs, cv=5)
            #add a branch to decTree with label A = v and subTree as subtree
    return decTree

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
        if (val_freq.has_key(record[target_attr])):
            val_freq[record[target_attr]] += 1.0
        else:
            val_freq[record[target_attr]] = 1.0

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
        if (val_freq.has_key(record[attr])):
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]] = 1.0

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        data_subset = [record for record in data if record[attr] == val]
        subset_entropy += val_prob * entropy(data_subset, target_attr)

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    return (entropy(data, target_attr) - subset_entropy)

print decisionTreeLearning(trainingSet, attributes)