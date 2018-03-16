import pandas as pnd
from sklearn import tree
from sklearn import datasets
from sklearn.datasets import load_iris
import graphviz

def decisionTreeLearning(examples, attributes, parentExamples):
    if examples is empty:
        return pluralityValue(parentExamples)
    elif all examples have the same classification:
        return classification
    elif attributes is empty:
        return pluralityValue(examples)
    else:
        A = argmax(importance(a, examples))
        decTree = tree.DecisionTreeClassifier()
        for each v in A:
            exs = e such that e is in examples and e.A = v
            subTree = decisionTreeLearning(exs, attributes - A, examples)
            add a branch to decTree with label A = v  and subTree as subtree
    return tree

def pluralityValue(set):
    return set