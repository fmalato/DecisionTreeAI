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


def trainTestSplit(dataset, start, end):
    """Reserve dataset.examples[start:end] for test; train on the remainder."""
    start = int(start)
    end = int(end)
    examples = dataset
    train = examples[:start] + examples[end:]
    val = examples[start:end]
    return train, val

def kFoldCrossValidation(k, examples, target, attributes, default):
    # The data set is divided into five parts by the trainTestSplit() function. For every iteration,
    # a different part of the data set is chosen as the validation set, while the others are used as
    # the training for the tree. Every criterion is validated with the same train/valid split. For each
    # iteration, then, a new tree is generated and tested.

    # This first part just initialize the variables.
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
        trainSet, validSet = trainTestSplit(examples, (len(examples)/k)*fold, (len(examples)/k)*(fold+1))
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

def getClassification(record, tree, attributes, default):
    # Given a record, it returns the classification for it.
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
        return getClassification(record, t, attributes, default)


def classify(tree, data, attributes, default):
    # For each record, appends a classification on a list and ultimately returns that list.
    data = data[:]
    classification = []

    for record in data:
        classification.append(getClassification(record, tree, attributes, default))

    return classification

def unknownDataTest(examples, attributes, target, times):
    # Tests on unknown data. The tree is trained and then tested on the examples attribute, which matches
    # with the test set.
    lista = [x[attributes.index(target)] for x in examples]
    default = dt.pluralityValue(lista)
    giniScores, entrScores, misclassScores = kFoldCrossValidation(times, examples, target, attributes, default)
    print
    print "Media giniScores: " + str(giniScores) + "  Media entrScores: " + str(entrScores) +\
          "  Media misClassScores: " + str(misclassScores)
