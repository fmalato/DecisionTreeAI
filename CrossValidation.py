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

def crossValidation(learner, size, k, examples):
    foldErrT = 0
    foldErrV = 0
    for fold in range(1, k):
        trainingSet, validationSet = partition(examples, fold, k)
        h = learner(size, trainingSet)
        foldErrT += errorRate(h, trainingSet)
        foldErrV += errorRate(h, validationSet)
    return foldErrT/k, foldErrV/k

def errorRate(heuristic, vector):
    errT = 0
    errV = 0
    for element in vector:
        if heuristic != element:
            errT += 1
            errV += 0.5
    return errT, errV

def classify(inputTree, attrs, testVec, classLabel):
    firstStr = list(inputTree)[0]
    secondDict = inputTree[firstStr]
    attrIndex = attrs.index(firstStr)
    alreadyDone = []
    default = float(dt.pluralityValue([x for x in testVec[len(testVec) - 1]]))
    for el in testVec:
        if el not in alreadyDone:
            if el[attrIndex] in secondDict:
                valueOfAttr = secondDict[el[attrIndex]]
            else:
                valueOfAttr = int(default)
            if isinstance(valueOfAttr, dict):
                alreadyDone = classify(valueOfAttr, attrs, [el], classLabel)
            else:
                classLabel.append(valueOfAttr)
                alreadyDone.append(el)
    return alreadyDone
