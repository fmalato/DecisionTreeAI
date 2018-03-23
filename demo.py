import pandas as pd
import numpy as np
import math
import csv

csvFile = csv.reader(file('/home/federico/Scrivania/Intelligenza Artificiale/Data Sets/shuttle.csv'), delimiter=",")
trainingSet = list(csvFile)
attributes = trainingSet[0]
del trainingSet[0]
A = np.zeros(7)

dictionarySet = {}
for j in range(len(attributes)):
    dictionarySet[attributes[j]] = [i[j] for i in trainingSet]
    k = 0
for j in attributes:
    A[k] = max(dictionarySet[j]) # print values, but not max values
    k += 1

print attributes
print
print trainingSet
print
print "trainingSet length: " + str(len(trainingSet))
print "trainingSet element length: " + str(len(trainingSet[0]))
print "attributes length: " + str(len(attributes))
print
print dictionarySet['Stability']
print
print "dictionary length: " + str(len(dictionarySet))
print
print A

