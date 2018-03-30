import numpy as np
import graphviz as gv
import math
from anytree import Node, RenderTree
from anytree.exporter import dotexporter as dotexp
import matplotlib.pyplot as plt
import copy
import networkx as netx
import csv
import DecisionTree as dt
import sys

sys.setrecursionlimit(1000000)

csvFile = csv.reader(file('/home/federico/Scrivania/Intelligenza Artificiale/Data Sets/playtennis.csv'), delimiter=",")
inputFile = list(csvFile)
attributes = inputFile[0]
targetAttr = attributes[len(attributes) - 1]
A = np.zeros(len(attributes))

examples = copy.deepcopy(inputFile)
del examples[0]

trainingSet = []
for line in examples:
    trainingSet.append(dict(zip(attributes, [datum.strip() for datum in line])))

decTree = dt.decisionTreeLearning(trainingSet, attributes, targetAttr)
dt.printTree(decTree, "")

