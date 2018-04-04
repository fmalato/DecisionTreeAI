import copy

import CrossValidation as cv

def walkdict(data, nodes):
    # Easily reach a specific node in a tree, given a dictionary
    for k, v in data.items():
        if isinstance(v, dict):
            walkdict(v, nodes)
        nodes.append((k,v))
        print("{0} : {1}".format(k, v))
    return nodes

def printTree(tree, str):
    # Prints the tree using a more readable format than the python's dict standard format
    if type(tree) == dict:
        print "%s%s" % (str, tree.keys()[0])
        for item in tree.values()[0].keys():
            print "%s\t--->\t%s" % (str, item)
            printTree(tree.values()[0][item], str + "\t\t\t")
    else:
        print "%s\t\t----->\t%s" % (str, tree)

def unique(lst):
    # Given a list, it returns a list without the redundant values.
    lst = lst[:]
    uniqueLst = []

    # Cycle through the list and add each value to the unique list only once.
    for item in lst:
        if uniqueLst.count(item) <= 0:
            uniqueLst.append(item)

    # Return the list with all redundant values removed.
    return uniqueLst

def dataDefinition(csvFile):
    inputFile = list(csvFile)
    # The first line of the .csv file is a list containing the attributes.
    attributes = inputFile[0]
    # That's why the target attribute should be in the last position in your .csv file. If you'd like to change,
    # then change the following line as well.
    targetAttr = attributes[len(attributes) - 1]
    # Makes a copy of the examples, removing the attributes.
    examples = copy.deepcopy(inputFile)
    del examples[0]
    # Just makes a copy of the 'examples' list, putting them in a dictionary.
    trainingSet = []
    for line in examples:
        trainingSet.append(dict(zip(attributes, [datum.strip() for datum in line])))
    # I provided ths due to the pluralityValue() function. I found out that makes it perform pretty well so... why not?
    lista = [x[attributes.index(targetAttr)] for x in examples]
    return attributes, targetAttr, examples, trainingSet, lista

def testDataset(csvFile):
    attributes, targetAttr, examples, trainingSet, lista = dataDefinition(csvFile)
    cv.unknownDataTest(examples, attributes, targetAttr, 5)