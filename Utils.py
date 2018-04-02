def walkdict(data, nodes):
    for k, v in data.items():
        if isinstance(v, dict):
            walkdict(v, nodes)
        nodes.append((k,v))
        print("{0} : {1}".format(k, v))
    return nodes

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