import numpy as np
import operator as op
import matplotlib 
import matplotlib.pyplot as plt

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    # use the local variable that stores the modules to call a function
    labels = ['A','A','B','B']
    return group,labels

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

def ReadData():
    filePath = "D:\GithubRepositories\MachineLearningPythonCodes\KnnAlgorithm\datingTestSet.txt"

    with open(filePath) as file:
        numberOfLines = len(file.readlines())
        propertyMatrix = np.zeros((numberOfLines,3))
        Label = []

    with open(filePath) as fr:
        index=0
        for line in fr.readlines():
            line = line.strip()
            listFromLine = line.split('\t')
            propertyMatrix[index,:] = listFromLine[0:3]
            Label.append(listFromLine[-1])
            index +=1
    return propertyMatrix,Label
    
def plotting(propMat,Labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(propMat[:,1], propMat[:,2])
    plt.title("Characteristics")
    plt.xlabel("Ice Cream Consumption")
    plt.ylabel("Video Game Duration")
    plt.show()

