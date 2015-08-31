
def classifyKnn(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    #.shape finds the dimension of the variable
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    #tile function repeats the array thus increasing the size
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    # sum with axis leads to calculation of sum rowwise
    distances = sqDistances**0.5
    sortedDistanceIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel=labels[sortedDistanceIndices[i]]
        # the label is stored here
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
        # Dictionary stores key:value , and adds value to 1
    sortedClassCount = sorted(classCount.iteritems(),key=op.itemgetter(1),reverse = True)
    #reverse sort
    return sortedClassCount[0][0]
    