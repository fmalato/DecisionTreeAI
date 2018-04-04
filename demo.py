import csv
import sys

import DecisionTree as dt
import Utils as util
# For great data sets, there could be a very high number of recursions while building the tree. I set it up to
# avoid sys errors.
sys.setrecursionlimit(1000000)
# For each data set, the function testDataset(csvFile) is called. It is declared on the Utils.py file and it
# defines the used variables. After that, it just calls the kFoldCrossValidation(), which starts the test.
# The data sets are chosen to represent some various scenarios: the first one has really messy data (and
# beacuse of that, it scores a bit worse than the others), the second one is a normal data set, while the
# third is nearly perfect (it's a bit smaller). Also, there's a fourth data set, not good for testing ( it
# has only 14 elements), but is good to print its tree and show it.
print "--------------------------------------------------"
print "----------- Dataset : Blood Donation -------------"
print "--------------------------------------------------"
print
csvFile1 = csv.reader(file('bloodDonation.csv'), delimiter=",")
util.testDataset(csvFile1)
print "--------------------------------------------------"
print "--------- Dataset : Nursery Classifire -----------"
print "--------------------------------------------------"
print
csvFile2 = csv.reader(file('nurseryClassifier.csv'), delimiter=",")
util.testDataset(csvFile2)
print "--------------------------------------------------"
print "------------- Dataset : Bankruptcy ---------------"
print "--------------------------------------------------"
print
csvFile3 = csv.reader(file('bankruptcy.csv'), delimiter=",")
util.testDataset(csvFile3)
print "--------------------------------------------------"
print "-------------- Graphic Tree Print ----------------"
print "--------------------------------------------------"
print
csvFile4 = csv.reader(file('playtennis.csv'), delimiter=",")
# Definition of the variables
attributes, targetAttr, examples, trainingSet, lista = util.dataDefinition(csvFile4)
default = dt.pluralityValue(lista)
dtree = dt.decisionTreeLearning(trainingSet, attributes, targetAttr, default, 'entropy')
# Printing a little tree, to show the way the decisionTreeLearning() function works.
util.printTree(dtree, "")
