import copy

import Heuristics as heur
import Utils as util

def decisionTreeLearning(trainingSet, attributes, targetAttr):

    localTraining = copy.deepcopy(trainingSet)
    localAttr = copy.deepcopy(attributes)
    values = [record[targetAttr] for record in trainingSet]

    default = pluralityValue(localTraining)

    if not localTraining or ((len(localAttr) - 1) <= 0):
        return default
    elif values.count(values[0]) == len(values):
        return values[0]
    else:
        maxGainAttr = chooseBestAttribute(localTraining, attributes, targetAttr)
        decTree = {maxGainAttr: {}}
        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for value in getValues(trainingSet, maxGainAttr):
            # Create a subtree for the current value under the "best" field
            subtree = decisionTreeLearning(getExamples(localTraining, maxGainAttr, value),
                                           [attr for attr in attributes if attr is not maxGainAttr],
                                           targetAttr)

            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            decTree[maxGainAttr][value] = subtree
    return decTree

def getExamples(data, attr, value):
    """
    Returns a list of all the records in <data> with the value of <attr>
    matching the given value.
    """
    data = data[:]
    finalList = []

    if not data:
        return finalList
    else:
        record = data.pop()
        if record[attr] == value:
            finalList.append(record)
            finalList.extend(getExamples(data, attr, value))
            return finalList
        else:
            finalList.extend(getExamples(data, attr, value))
            return finalList

def getValues(data, attribute):
    data = data[:]
    return util.unique([record[attribute] for record in data])

def arraySubtraction(array, attribute):
    del array[attribute]
    return array

def pluralityValue(data):
    lst = data[:]
    highestFreq = 0
    mostFreq = None

    for val in util.unique(lst):
        if lst.count(val) > highestFreq:
            mostFreq = val
            highestFreq = lst.count(val)

    return mostFreq

def chooseBestAttribute(data, attributes, target):
    data = data[:]
    bestGain = 0.0
    bestAttr = None

    for attr in attributes:
        gainAttr = heur.gain(data, attr, target)
        if (gainAttr >= bestGain and attr != target):
            bestGain = gainAttr
            bestAttr = attr

    return bestAttr

