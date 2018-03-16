import pandas as pd
import numpy as np
import graphviz as gv
from sklearn import tree
from sklearn import datasets
from sklearn.datasets import load_iris

trainingSet = pd.read_csv('/home/federico/Scrivania/Intelligenza Artificiale/Data Sets/winequality-white.csv', sep=';')

def decisionTreeLearning(trainingSet, attributes):

    if all(x == x for x in trainingSet['quality']):
        return trainingSet[0, 'quality']
    if not attributes:
        return pluralityValue(trainingSet)
    else:
        A = np.argmax(attributes, importance(a, examples))
        decTree = tree.DecisionTreeClassifier(examples, attributes)
        for each v in A:
            exs = e such that e is in examples and e.A = v
            subTree = decisionTreeLearning(exs, attributes - A, examples)
            add a branch to decTree with label A = v  and subTree as subtree
    return decTree

def pluralityValue(set):
    return set

def importance(setElement, set):
    return setElement