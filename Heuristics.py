import math
import numpy as np
from collections import defaultdict

import Utils as util

def entropy(data, targetAttr):
    """
    Calculates the entropy of the given data set for the target attribute.
    """
    valFreq = {}
    dataEntropy = 0.0

    # Calculate the frequency of each of the values in the target attr
    for record in data:
        if (valFreq.has_key(record[targetAttr])):
            valFreq[record[targetAttr]] += 1.0
        else:
            valFreq[record[targetAttr]] = 1.0

    # Calculate the entropy of the data for the target attribute
    for freq in valFreq.values():
        dataEntropy += (-freq/len(data)) * math.log(freq/len(data), 2)

    return dataEntropy


def gainEntr(data, attr, targetAttr):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """
    valFreq = {}
    subsetEntropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for record in data:
        if (valFreq.has_key(record[attr])):
            valFreq[record[attr]] += 1.0
        else:
            valFreq[record[attr]] = 1.0

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in valFreq.keys():
        valProb = valFreq[val] / sum(valFreq.values())
        dataSubset = [record for record in data if record[attr] == val]
        subsetEntropy += valProb * entropy(dataSubset, targetAttr)

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    return (entropy(data, targetAttr) - subsetEntropy)

def gini(data, attr, targetAttr):
    valFreq = {}
    giniInd = 0.0

    for record in data:
        if (valFreq.has_key(record[attr])):
            valFreq[record[attr]] += 1.0
        else:
            valFreq[record[attr]] = 1.0
    for val in valFreq.keys():
        valProb = valFreq[val] / sum(valFreq.values())
        giniInd += valProb*(1 - valProb)
    return giniInd

def gainGini(data, attr, targetAttr):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """
    valFreq = {}
    subsetEntropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for record in data:
        if (valFreq.has_key(record[attr])):
            valFreq[record[attr]] += 1.0
        else:
            valFreq[record[attr]] = 1.0

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in valFreq.keys():
        valProb = valFreq[val] / sum(valFreq.values())
        dataSubset = [record for record in data if record[attr] == val]
        subsetEntropy += valProb * gini(dataSubset, attr, targetAttr)

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    return (gini(data, attr, targetAttr) - subsetEntropy)

def misclassificationError(data, attr, target):
    valFreq = {}

    for record in data:
        if (valFreq.has_key(record[attr])):
            valFreq[record[attr]] += 1.0
        else:
            valFreq[record[attr]] = 1.0
    maxValue = max(valFreq.values())
    return 1.0 - maxValue / len(data)

def gainMisclass(data, attr, targetAttr):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """
    valFreq = {}
    subsetEntropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for record in data:
        if (valFreq.has_key(record[attr])):
            valFreq[record[attr]] += 1.0
        else:
            valFreq[record[attr]] = 1.0

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in valFreq.keys():
        valProb = valFreq[val] / sum(valFreq.values())
        dataSubset = [record for record in data if record[attr] == val]
        subsetEntropy += valProb * misclassificationError(dataSubset, attr, targetAttr)

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    return (misclassificationError(data, attr, targetAttr) - subsetEntropy)