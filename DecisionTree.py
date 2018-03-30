import numpy as np
import graphviz as gv
import math
import anytree
from anytree.exporter import dotexporter as dotexp
import matplotlib.pyplot as plt
import copy
import networkx as netx
import csv
from anytree.exporter import DotExporter

def decisionTreeLearning(trainingSet, attributes, targetAttr):

    localTraining = copy.deepcopy(trainingSet)
    localAttr = copy.deepcopy(attributes)
    values = [record[targetAttr] for record in trainingSet]

    #vals = [record for record in dictionarySet[targetAttr]]
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
    rtn_lst = []

    if not data:
        return rtn_lst
    else:
        record = data.pop()
        if record[attr] == value:
            rtn_lst.append(record)
            rtn_lst.extend(getExamples(data, attr, value))
            return rtn_lst
        else:
            rtn_lst.extend(getExamples(data, attr, value))
            return rtn_lst

def getValues(data, attribute):
    data = data[:]
    return unique([record[attribute] for record in data])

def arraySubtraction(array, attribute):
    del array[attribute]
    return array

def pluralityValue(data):
    lst = data[:]
    highestFreq = 0
    mostFreq = None

    for val in unique(lst):
        if lst.count(val) > highestFreq:
            mostFreq = val
            highestFreq = lst.count(val)

    return mostFreq

def chooseBestAttribute(data, attributes, target):
    data = data[:]
    best_gain = 0.0
    best_attr = None

    for attr in attributes:
        gainAttr = gain(data, attr, target)
        if (gainAttr >= best_gain and attr != target):
            best_gain = gainAttr
            best_attr = attr

    return best_attr

def unique(lst):
    """
    Returns a list made up of the unique values found in lst.  i.e., it
    removes the redundant values in lst.
    """
    lst = lst[:]
    uniqueLst = []

    # Cycle through the list and add each value to the unique list only once.
    for item in lst:
        if uniqueLst.count(item) <= 0:
            uniqueLst.append(item)

    # Return the list with all redundant values removed.
    return uniqueLst


def entropy(data, target_attr):
    """
    Calculates the entropy of the given data set for the target attribute.
    """
    val_freq = {}
    data_entropy = 0.0

    # Calculate the frequency of each of the values in the target attr
    for record in data:
        if (val_freq.has_key(record[target_attr])):
            val_freq[record[target_attr]] += 1.0
        else:
            val_freq[record[target_attr]] = 1.0

    # Calculate the entropy of the data for the target attribute
    for freq in val_freq.values():
        data_entropy += (-freq/len(data)) * math.log(freq/len(data), 2)

    return data_entropy


def gain(data, attr, target_attr):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """
    val_freq = {}
    subset_entropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for record in data:
        if (val_freq.has_key(record[attr])):
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]] = 1.0

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        data_subset = [record for record in data if record[attr] == val]
        subset_entropy += val_prob * entropy(data_subset, target_attr)

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    return (entropy(data, target_attr) - subset_entropy)

def walkdict(data, nodes):
    for k, v in data.items():
        if isinstance(v, dict):
            walkdict(v, nodes)
        nodes.append((k,v))
        print("{0} : {1}".format(k, v))
    return nodes

def drawDecisionTree(dt):
    nodes = []
    graphNodes = []
    nodesList = list(walkdict(dt, nodes))
    #parent = anytree.Node(nodesList[len(nodesList) - 1])
    graph = netx.Graph()
    realGraph = copy.deepcopy(graph)
    for attr in nodesList:
        graph.add_node(attr[0], label=attr[0])
        realGraph.add_node(attr[0], label=attr[0])
    for attr in nodesList:
        if attr[1] == 'yes':
            graph.add_node('Yes', label='Yes')
            realGraph.add_node('Yes', label='Yes')
    for attr in nodesList:
        if attr[1] == 'no':
            graph.add_node('No', label='No')
            realGraph.add_node('No', label='No')
    for node in graph:
        realGraph.add_edge(node, (x[1] for x in nodesList if x[0] == node.name))
    return realGraph

def printTree(tree, str):
    """
    This function recursively crawls through the d-tree and prints it out in a
    more readable format than a straight print of the Python dict object.  
    """
    if type(tree) == dict:
        print "%s%s" % (str, tree.keys()[0])
        for item in tree.values()[0].keys():
            print "%s\t--->\t%s" % (str, item)
            printTree(tree.values()[0][item], str + "\t\t\t")
    else:
        print "%s\t\t----->\t%s" % (str, tree)
