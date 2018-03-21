import pandas as pd
import numpy as np
import math
import csv

csvFile = csv.reader(file('/home/federico/Scrivania/Intelligenza Artificiale/Data Sets/winequality-white.csv'), delimiter=";")
trainingSet = list(csvFile)
attributes = trainingSet[0]
del trainingSet[0]

dictionarySet = {}
for j in range(len(attributes)):
    dictionarySet[attributes[j]] = [i[j] for i in trainingSet]
print attributes
print
print trainingSet
print
print "trainingSet length: " + str(len(trainingSet))
print "trainingSet element length: " + str(len(trainingSet[0]))
print "attributes length: " + str(len(attributes))
print
print dictionarySet

