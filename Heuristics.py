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


def gain(data, attr, targetAttr):
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


def giniImpurity(examples, attributes):
    total = len(examples)
    counts = {}
    imp = 0
    for item in attributes:
        counts.setdefault(str(item), 0)
        imp = 0
        for j in examples:
            f1 = float(counts[str(j)]) / total
            for k in examples:
                if j == k:
                    continue
                f2 = float(counts[str(k)]) / total
                imp += f1 * f2
    return imp

def giniIndex(examples, attributes):
    total = len(examples)
    imp = []
    gini = 1
    dictionary = {}
    for attr in attributes:
        dictionary[attr] = [ex[attributes.index(attr)] for ex in examples]
        x = float(len(util.unique(dictionary[attr])))
        actualGini = 1 - (x / total)
        if actualGini < gini:
            gini = actualGini
            index = attributes.index(attr)
    return gini, index

def best_splitV6(feature_values, labels):
    # training for each node/feature determining the threshold
    feature_values, labels = np.array(feature_values), np.array(labels)

    impurity = []
    possible_thresholds = np.unique(feature_values)

    num_labels = labels.size

    # the only relevant possibilities for a threshold are the feature values themselves except the lowest value

    for threshold in possible_thresholds:
        # split node content based on threshold
        # to do here: what happens if len(right) or len(left) is zero
        selection = feature_values>=threshold

        right = labels[selection]
        left = labels[~selection]

        num_right = right.size

        # compute distribution of labels for each split
        _ , right_distribution = np.unique(right, return_counts=True)
        _ , left_distribution = np.unique(left, return_counts=True)

        # compute impurity of split based on the distribution
        gini_right = 1 - np.sum((np.array(right_distribution) / num_right) ** 2)
        gini_left = 1 - np.sum((np.array(left_distribution) / (num_labels-num_right)) ** 2)

        # compute weighted total impurity of the split
        gini_split = (num_right * gini_right + (num_labels-num_right) * gini_left) / num_labels

        impurity.append(gini_split)


    # returns the threshold with the highest associated impurity value --> best split threshold
    return possible_thresholds[np.argmin(impurity)]