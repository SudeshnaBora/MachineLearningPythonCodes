import numpy as np
import operator as op
import matplotlib 
import matplotlib.pyplot as plt



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
         SmallIndexTotal = 0
         NoIndexTotal = 0
         LargeIndexTotal =0
         for line in file.readlines():
             line = line.strip()
             listFromLine = line.split('\t')
             if listFromLine[-1]== 'smallDoses':
                 SmallIndexTotal +=1
             elif listFromLine[-1]== 'largeDoses':
                 LargeIndexTotal += 1
             else:
                 NoIndexTotal +=1
         SmallMat = np.zeros((SmallIndexTotal,3))
         NoMat = np.zeros((NoIndexTotal,3))
         LargeMat = np.zeros((LargeIndexTotal,3))
         Labels = []
     with open(filePath) as fid:
         numberOfLines = len(fid.readlines())
         propertyMatrix = np.zeros((numberOfLines,3))

     with open(filePath) as fr:
         SmallIndex=0
         NoIndex=0
         LargeIndex=0
         Index = 0
         for line in fr.readlines():
             line = line.strip()
             listFromLine = line.split('\t')
             propertyMatrix[Index,:] = listFromLine[0:3]
             Index += 1
             if listFromLine[-1]== 'smallDoses':
                 SmallMat[SmallIndex,:]= listFromLine[0:3]
                 SmallIndex +=1
             elif listFromLine[-1]== 'largeDoses':
                 LargeMat[LargeIndex,:]= listFromLine[0:3]
                 LargeIndex +=1
             else:
                 NoMat[NoIndex,:]= listFromLine[0:3]
                 NoIndex +=1
             Labels.append(listFromLine[-1])
     return propertyMatrix,Labels
    
def plotting(NoMat,SmallMat,LargeMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    a1=ax.scatter(NoMat[:,0], NoMat[:,1],c='b',label='Didn\'t like')
    a2=ax.scatter(SmallMat[:,0], SmallMat[:,1],c='g',label='Small doses')
    a3=ax.scatter(LargeMat[:,0], LargeMat[:,1],c='r',label='Large doses')
    plt.title("Characteristics")
    plt.xlabel("Flying Miles")
    plt.ylabel("Video Game Duration")
    plt.legend(handles=[a1,a2,a3],loc = 'upper left')
    plt.show()
    
def AutoNorm(DataSet):
    minValues = DataSet.min(0) #columnwise max and min
    maxValues = DataSet.max(0)
    ranges = maxValues - minValues #columnwise range
    normDataSet = np.zeros(np.shape(DataSet)) #make a matrix of 0 to store normalised data
    normDataSet = DataSet - minValues
    normDataSet = normDataSet/ranges
    return normDataSet,ranges,minValues
    
def DatingClassTest():
    ratio = 0.10
    datingDataMat,Labels = ReadData()
    normDataMat = AutoNorm(datingDataMat)
    m = normDataMat.shape[0]
    TestLim = int(m*ratio)
    errorCount = 0.0
    for i in range(TestLim):
        ClassfierResult = classifyKnn(normDataMat[i,:],normDataMat[TestLim:m,:],Labels[TestLim:m],3)
        print "The classfier result is %s" %ClassfierResult
        print "The real result is %s" %Labels[i]
        if (ClassfierResult != Labels[i]):
            errorCount +=1.0
    print "The Error is %f" %(errorCount/float(TestLim))
    
def KnnCaller():
    time = float(raw_input('Time spent in playing video games (in %)'))
    iceCream = float(raw_input('Amount of ice cream consumed'))
    flier = float(raw_input('miles flied'))
    datingDataMat,Labels = ReadData()
    normDataMat,ranges,minValues = AutoNorm(datingDataMat)
    inArray = np.array([flier,time,iceCream])
    
    result = classifyKnn((inArray-minValues)/ranges,normDataMat,Labels,3)
    print result    

