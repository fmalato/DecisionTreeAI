import random as rand
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

def kFoldCrossValidation(k, examples, target, attributes, default):
    foldErr = 0
    score = 0.0
    giniScores = []
    entropyScores = []
    missclassScores = []
    criteria = ['gini', 'entropy', 'missclass']
    rand.shuffle(examples)
    for fold in range(k):
        print "####################"
        print "### iterazione " + str(fold+1) + " ###"
        print "####################"
        print
        trainSet, validSet = train_test_split(examples, (len(examples)/k)*fold, (len(examples)/k)*(fold+1))
        training = []
        for line in trainSet:
            training.append(dict(zip(attributes, [datum.strip() for datum in line])))
        for crit in criteria:
            print "Criterio: " + crit
            tree = dt.decisionTreeLearning(training, attributes, target, default, crit)
            classification = classify(tree, validSet, attributes, default)
            x = 0
            for el in validSet:
                if classification[x] == el[attributes.index(target)]:
                    score += 1.0
                x += 1
            print "Round score: " + str(score/(len(validSet)))
            if crit == 'gini':
                giniScores.append(score/len(validSet))
            elif crit == 'entropy':
                entropyScores.append(score/len(validSet))
            elif crit == 'missclass':
                missclassScores.append(score/len(validSet))
            score = 0.0
    return (sum(giniScores))/k, (sum(entropyScores))/k, (sum(missclassScores))/k

def errorRateT(classification, vector, target, attributes):
    errT = 0
    for element in vector:
        if classification[vector.index(element)] != element[attributes.index(target)]:
            errT += 1
    return errT

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
