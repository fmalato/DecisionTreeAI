def walkdict(data, nodes):
    for k, v in data.items():
        if isinstance(v, dict):
            walkdict(v, nodes)
        nodes.append((k,v))
        print("{0} : {1}".format(k, v))
    return nodes

"""def drawDecisionTree(dt):
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
"""
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