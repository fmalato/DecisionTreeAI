import copy

import Heuristics as heur
import Utils as util

def decisionTreeLearning(trainingSet, attributes, targetAttr, default, criterion='entropy'):
    # At first, the function makes some deep copies in order to avoid mess during the various recursions.
    # This copies may slow down the process a bit, but it's worth the effort.
    localTraining = copy.deepcopy(trainingSet)
    localAttr = copy.deepcopy(attributes)
    values = [record[targetAttr] for record in trainingSet]
    # Some default cases: if there's nothing to split, it just doesn't.
    if not localTraining or ((len(localAttr) - 1) <= 0):
        return default
    elif values.count(values[0]) == len(values):
        return values[0]
    else:
        maxGainAttr = chooseBestAttribute(localTraining, attributes, targetAttr, criterion)
        decTree = {maxGainAttr: {}}
        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field.
        for value in getValues(trainingSet, maxGainAttr):
            # Create a subtree for the current value under the "best" field.
            subtree = decisionTreeLearning(getExamples(localTraining, maxGainAttr, value),
                                           [attr for attr in attributes if attr is not maxGainAttr],
                                           targetAttr, default, criterion)

            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            decTree[maxGainAttr][value] = subtree
    return decTree

def getExamples(data, attr, value):
    # Returns a list of all the records in <data> with the value of <attr>
    # matching the given value.
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
    # Simply removes a column for a list/dictionary.
    del array[attribute]
    return array

def pluralityValue(data):
    # Returns the most frequent value in a given list.
    lst = data[:]
    highestFreq = 0
    mostFreq = None

    for val in util.unique(lst):
        if lst.count(val) > highestFreq:
            mostFreq = val
            highestFreq = lst.count(val)

    return mostFreq

def chooseBestAttribute(data, attributes, target, criterion):
    # This function chooses the heuristic that must be applied in order to fulfill the
    # user's request. For each criterion there's a different call. Otherwise, it throws
    # an exception which stops the program execution.
    data = data[:]
    bestGain = 0.0
    bestAttr = None

    for attr in attributes:
        try:
            if criterion == 'gini':
                gainAttr = heur.gainGini(data, attr, target)
            elif criterion == 'entropy':
                gainAttr = heur.gainEntr(data, attr, target)
            elif criterion == 'missclass':
                gainAttr = heur.gainMisclass(data, attr, target)
            else:
                raise Exception('No criterion with such name.')
        except Exception as ex:
            print ex.args
        if (gainAttr >= bestGain and attr != target):
            bestGain = gainAttr
            bestAttr = attr

    return bestAttr

