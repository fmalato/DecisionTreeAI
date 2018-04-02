import numpy as np
import copy

import DecisionTree as dt

def partition(vector, fold, k):
    setSize = len(vector)
    start = (setSize/k)*(fold)
    end = (setSize/k)*(fold+1)
    validationVector = copy.deepcopy(vector[start:end])
    training = copy.deepcopy(vector[0:start])
    return training, validationVector


def train_test_split(dataset, start, end):
    """Reserve dataset.examples[start:end] for test; train on the remainder."""
    start = int(start)
    end = int(end)
    examples = dataset
    train = examples[:start] + examples[end:]
    val = examples[start:end]
    return train, val

def kFoldCrossValidation(tree, k, examples, target, attributes, default):
    foldErrT = 0
    foldErrV = 0
    for fold in range(0, k - 1):
        # Must change last two parameters in the function call...
        trainSet, validSet = train_test_split(examples, (fold*len(examples))/k, ((fold + 1)*len(examples))/k)
        classificationTrain = classify(tree, trainSet, attributes, default)
        classification = classify(tree, validSet, attributes, default)
        for element in trainSet:
            foldErrT += errorRateT(classificationTrain, trainSet, target, attributes)
        for element in validSet:
            foldErrV += errorRateV(classification, validSet, target, attributes)
    return foldErrT / k, foldErrV / k

def errorRateT(classification, vector, target, attributes):
    errT = 0
    for element in vector:
        if classification[vector.index(element)] != element[attributes.index(target)]:
            errT += 1
    return errT

def errorRateV(classification, vector, target, attributes):
    errV = 0
    for element in vector:
        if classification[vector.index(element)] != element[attributes.index(target)]:
            errV += 0.5
    return errV

def get_classification(record, tree, attributes, default):
    """
    This function recursively traverses the decision tree and returns a
    classification for the given record.
    """
    # If the current node is a string, then we've reached a leaf node and
    # we can return it as our answer
    if type(tree) == type("string"):
        return tree

    # Traverse the tree further until a leaf node is found.
    else:
        attr = tree.keys()[0]
        if record[attributes.index(attr)] in tree[attr]:
            t = tree[attr][record[attributes.index(attr)]]
        else:
            t = default
        return get_classification(record, t, attributes, default)


def classify(tree, data, attributes, default):
    """
    Returns a list of classifications for each of the records in the data
    list as determined by the given decision tree.
    """
    data = data[:]
    classification = []

    for record in data:
        classification.append(get_classification(record, tree, attributes, default))

    return classification
