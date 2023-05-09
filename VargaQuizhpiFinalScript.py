import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.datasets import make_blobs
import random
from collections import Counter

# Function to make a file with nPoints, nDims, between minVal and maxVal.
# Chooses a random number of centers and creates points normally distributed around centers.
def makePointsFile(nPoints, nDims, minVal, maxVal, filename):
        middleVal = np.mean([abs(maxVal), abs(minVal)])
        nCenters = np.random.randint(1,10)
        evenDiv = int(np.floor(nPoints/nCenters))
        extraPoints = nPoints-(nCenters*evenDiv)
        ptz = np.zeros((nDims,nPoints))
        for k in range(nCenters):
            np.random.seed()
            center = np.random.randint(np.floor(2*minVal/3),np.floor(2*maxVal/3))
            stupidList = [[np.random.normal(center, middleVal/8) for j in range(evenDiv)] for n in range(nDims)]
            trifecta = np.array(stupidList)
            ptz[:,k*evenDiv:(k+1)*evenDiv] = trifecta
        if extraPoints:
            ptz[:,-extraPoints:]=[[np.random.normal(center, middleVal/8) for i in range(extraPoints)] for n in range(nDims)]
        ptzUse = ptz.T
  
        with open(filename, 'w') as curPoints:
            for point in ptzUse:
                ptz2write = np.array2string(point) + '\n'
                curPoints.write(ptz2write)
 
 
 
#main class which contains KMeans, KNN, and related attribuites and functions
class Classifier:
    def __init__(self, points, k, verboseMode, tolerance=.05, *minmaxVal): #add input for data set, take vectors from random gaussian distribution and compare
        if minmaxVal and isinstance(points, int):
            self.__maxVal = max(minmaxVal)
            self.__minVal = min(minmaxVal)
        elif isinstance(points, int):
            raise ValueError('Must have min and max values to generate points')
        else:
            self.__maxVal = np.max(points)
            self.__minVal = np.min(points)
        self.k = k
        self.verbosity = verboseMode
        self.n = points
        self.tolerance = tolerance
 
    @property #getter for numPoints
    def n(self):
        return self._n
 
    @n.setter #setter for _points and _n - must be positive int or list
    def n(self, newPoints):
        if isinstance(newPoints, int) and newPoints > 0:
            self._n = newPoints
            self._numDims = int(input('How many dimensions?'))
            self.genData = True
        elif isinstance(newPoints, list):
            self._points = np.array(newPoints)
            self._points = newPoints
            nDim, x = self._points.shape
            if x < nDim:
                self._points.T
            self._n = max(x,nDim)
            self._numDims = min(x,nDim)
            self.genData = False
        elif isinstance(newPoints, np.ndarray):
            self._points = newPoints
            nDim, x = self._points.shape
            if x < nDim:
                self._points.T
            self._n = max(x,nDim)
            self._numDims = min(x,nDim)
            self.genData = False
        else:
            raise ValueError('# of points must be positive int')
 
    @property #getter for num of cluster heads
    def k(self):
        return self._k
 
    @k.setter #setter for num of cluster heads - must be more than 1 for now
    def k(self, clusterhead):
        if isinstance(clusterhead, int) and clusterhead > 1:
           self._k = clusterhead
        else:
            raise ValueError('# of clusterheads must be more than 1!')
 
 
    def _getPoints(self, numPoints):
        #Make an array of random points between maxVal and minVal
        plt.figure()
        middleVal = np.mean([abs(self.__maxVal), abs(self.__minVal)])
        evenDiv = int(np.floor(self.n/self.k))
        extraPoints = self.n-(self.k*evenDiv)
        pointz = np.zeros((self._numDims,self.n))
        clustersList = []
        for k in range(self.k):
            np.random.seed()
            center = np.random.randint(np.floor(self.__minVal*.8),np.floor(self.__maxVal*.8))
            trifecta = np.array([[np.random.normal(center, middleVal/10) for j in range(self._numDims)] for n in range(evenDiv)]).T
            pointz[:,k*evenDiv:(k+1)*evenDiv] = trifecta
            clusterSet = set({})
            for j in trifecta.T:
                clusterSet.add(tuple(j))
            clustersList.append(clusterSet)
        if extraPoints:
            pointz[:,-extraPoints:]=[[np.random.normal(0, middleVal/8) for i in range(extraPoints)] for n in range(self._numDims)]
        self._trueAsst = clustersList
        if self.verbosity == 1:
            print(f'data: \n{pointz}')
        return pointz
 
    def _firstClusterhead(self):
        # This function creates an array for the coordinates of clusterheads.
        # Initizialized as random points chosen from self._points
      
        #initialize clusterhead array
        self.clusterheads = np.zeros(shape = (self._numDims, self.k))
        for m in range(self.k):
            self.clusterheads[:,m] = self._points[:,np.random.randint(self.n)]
        #2d plotting
        if self._numDims == 2:
            if self.verbosity == 1:
                plt.figure()
                plt.axis([self.__minVal*1.5, self.__maxVal*1.5,self.__minVal*1.5, self.__maxVal*1.5])
                headCounter = 0
                for heads in self.clusterheads.T:
                    plt.plot(heads[0], heads[1], '+', label= f'Clusterhead {headCounter}')
                    headCounter+=1       
      
        #forms the assignment matrix - random clusterhead placement comes later
        self.assignmentMat = np.zeros((self._numDims+1, self.n))
        self.assignmentMat[1:,:] = np.array([self._points[0:,l] for l in range(self.n)]).T
      
    def _formCluster(self):
        #compares each point to a clusterhead
        distMat = np.zeros((self.n,self.k))
      
        #iterate points
        for i in range(self.n):
            pointDist = np.array([])
            for j in range(self.k): #for each clusterhead check the distance
                pointDist = np.append(pointDist, self._distance(self._points[:,i], self.clusterheads[:,j]))
            for ind in np.where(pointDist == pointDist.min()):
                if ind.any():
                    self.assignmentMat[0, i] = ind[0]
          
            distMat[i,:] = pointDist
      
        self.__distances = distMat
 
    def _updateClusterhead(self):
        #Going to plot the points
        if self._numDims == 2:
            # Plotting in 2d
            if self.verbosity == 1:
                plt.figure()
                plt.axis([self.__minVal-5, self.__maxVal+5,self.__minVal-5, self.__maxVal+5])
 
    # Saving the last cluster heads to compare for end condition of K-means
        self.lastClusterheads = np.ndarray.copy(self.clusterheads)
        for k in range(self.k):
            if k in self.assignmentMat[0,:]:
                self.clusterheads[:,k] = np.mean(self.assignmentMat[1:, self.assignmentMat[0] == k], axis = 1)
            else:
                print(f'cluster head: {self.clusterheads[:,k]} has no points assigned')
            if self.verbosity == 1:
                #2d plot
                if self._numDims == 2:
                    plt.plot(self.assignmentMat[1, self.assignmentMat[0] == k], self.assignmentMat[2, self.assignmentMat[0] == k], '.')
                print(f'new cluster heads: \n{self.clusterheads}\n last heads: \n{self.lastClusterheads}')
      
      
        if self._numDims == 2:
            headCounter = 0
            #2d plotting clusterheads
            if self.verbosity == 1:
                for heads in self.clusterheads.T:
                    plt.plot(heads[0], heads[1], '+', label= f'Clusterhead {headCounter}')
                    headCounter+=1
 
    def _distance(self, item1, item2):
        #_distance method takes input of two iterables each with numDim values
        #squared distance formula
        squareNum = np.square(item2 - item1)
        findsum = np.sum(squareNum)
        return findsum
 
    def readPointsFile(filename):
        if isinstance(filename, str):
            with open(filename, 'r') as readPoints:
                pointsList = readPoints.read()
                curPattern = re.compile(r'[\[\(]?([0-9\.-]{6,12})[ ,]{1,3}([0-9\.-]{6,12})[\]\)]?', flags=re.MULTILINE)
                curMatch = curPattern.finditer(pointsList)
                dataIn = []
                for match in curMatch:
                    newPt = [float(match.group(1)), float(match.group(2))]
                    dataIn.append(newPt)
                dataIn = np.array(dataIn).T
                return dataIn
        else:
            raise ValueError('File name must be a string.')
 
    def seePlot(self):
        plt.legend()
        plt.xlabel('X-values')
        plt.ylabel('Y-Values')
        plt.title('K-means results')
        plt.show()
  
    def errorCheck(self):
        numError = 0
        for k in range(self.k):
            clusAsst = set({})
            for point in self.assignmentMat[1:, self.assignmentMat[0] == k].T:
                clusAsst.add(tuple(point))
            for j in range(self.k):
                #print(f'{self._trueAsst[j]}')
                if self._trueAsst[j].intersection(clusAsst):
                    if self._trueAsst[j].intersection(clusAsst) != self._trueAsst[j]:
                        numError += len(self._trueAsst[j].difference(clusAsst))
                        #print(f'difference is {self._trueAsst[j].difference(clusAsst)}\nTruth is {self._trueAsst[j]}')
        
        return numError/self.k

    def KMeans(self):
        if self.genData == True:
            self._points = self._getPoints(self.n)
        for j in range(10):
            self._firstClusterhead()
            self._formCluster()
            self._updateClusterhead()
            numIter = 1
            iterList = []
            if self.genData == True:
                errorList = []
            while self._distance(self.clusterheads, self.lastClusterheads) > self.tolerance:
                print(f'distance between old and new heads: {self._distance(self.clusterheads, self.lastClusterheads):.2f}')
                self._formCluster()
                self._updateClusterhead()
                numIter+=1
                # Error checking
                if self.genData == True:
                    nError = self.errorCheck()
                    print(f'error was {nError/self.n}')
                    errorList.append(nError/self.n)
                if self.verbosity == 1 and self._numDims == 2:
                    self.seePlot()
            iterList.append(numIter)
            print(f'Finished try! Distance between old and new heads: {self._distance(self.clusterheads, self.lastClusterheads):f}\n')
                
        if self.genData == True:
            return iterList, errorList 
        else:
            return iterList    
 
    
    #KNN methods
    def makeDataSet(self, numSamples, numClusters, standardDev,nFeatures):
      # X : array of shape [n_samples, n_features]
      # y : array of shape [n_samples]
        X,y= make_blobs(
          # number of points divided in clusters
          n_samples = numSamples,
    
            # how many features do u want the sample to have (x,y)
            # basically how complex is the data
            n_features = nFeatures,
            # how many clusters
            centers = numClusters,
    
            center_box =(self.__minVal,self.__maxVal),
            # standard deviation of the cluster
            # smaller the standard deviation the better
            cluster_std = standardDev,)
        dataSet = []
        data = []
        if self.verbosity == 1:
            plt.figure()
            plt.scatter(X[:,0],
                        X[:,1],
                        c = y)
            plt.title('KNN Points')
        #print(X.shape)
        for vv in range(len(y)):
            data = [X[vv],y[vv]]
            dataSet.append(data)
        return dataSet
    
    def createTrainingAndTestingSet(self,dataSet):
        # Creates a training set that is 80% of the data and a testing set that is the remaing 20%
        TrainingSet = dataSet
        TestingSet = []
        dataSetNum = len(dataSet)
        for vv in range(round(len(dataSet)*.2)):
            randomIndex = random.randrange(0,dataSetNum)
            TestingSet.append(TrainingSet[randomIndex])
            TrainingSet.pop(randomIndex)
            dataSetNum -= 1
        return TrainingSet, TestingSet
     
    def distanceCalculator(self,dataP1, dataP2):
        # Calculates the distance between two points given any number of dimensions
        point1 = np.array(dataP1[:-1])
        point2 = np.array(dataP2[0])
        sqaureNum = np.square(point1-point2)
        findsum = np.sum(sqaureNum)
        euclideanDistance = np.sqrt(findsum)
        return euclideanDistance

    def sortList(self,dataDistanceList):
        # sorts the the list of points with distances from smallest distance to largest distance
        return(sorted(dataDistanceList, key = lambda x : x[1]))
    
    def randomTestingPoint(self,TrainingAndTestingSet):
        # generates a random point to test from the training set
        TrainingSet = TrainingAndTestingSet[1]
        randomIndex = random.randrange(0,len(TrainingAndTestingSet))
        testPoint = TrainingSet[randomIndex]
        TrainingSet.pop(randomIndex)
        return testPoint
    
    def findNeigbors(self,TestSet,randomPoint, k):
        # find the K nerest neigbors from testing point to all points in the traing set
        # outputs the nearest neigbor
        distanceBetweenP = []  
        for vv in range(len(TestSet)):
            distance = self.distanceCalculator(TestSet[vv],randomPoint)
            distanceBetweenP.append([TestSet[vv],distance])
        distanceInOrder = self.sortList(distanceBetweenP)
        nearestK = []
        for vv in range(k):
            nearestK.append(distanceInOrder[vv])
        return nearestK
    
    def findMostCommonneigbor(self,ListNeigborGroups):
        countFrequen = Counter(ListNeigborGroups)
        mostFreq = countFrequen.most_common(1)[0][0]
        return mostFreq  
    
    def ListOfBelongingGroup(self,findKNeighbors):
        ListNeigborGroups = []
        for vv in range(len(findKNeighbors)):
            group = findKNeighbors[vv][0][1]
            ListNeigborGroups.append(group)  
        belongingGroup = self.findMostCommonneigbor(ListNeigborGroups) 
        return belongingGroup
    
    def CheckIfCorrect(self,TestPointWithGroup):
        realGroup = TestPointWithGroup[0][1]
        KNNSORTEDGROUP = TestPointWithGroup[1]
        if realGroup == KNNSORTEDGROUP:
            return True
        else:
            return False
        
    def performKNN(self,numClusters):
        # creates a Standard Devation of 10% of the amount of data
        standardDev = round(self.n * .01)
        timesCorrect = 0
        timesInccorect = 0
        officialDataSet = self.makeDataSet(self.n,numClusters,standardDev,self._numDims)
        # Repeated 10 times
        for vv in range(10):
            TrainAndTest = self.createTrainingAndTestingSet(officialDataSet)
            TrainingSet = TrainAndTest[0]
            TestSet = TrainAndTest[1]
            RandomPoint = self.randomTestingPoint(TrainAndTest)
            findKNeigbors = self.findNeigbors(TrainingSet,RandomPoint,self.k)
            GroupPointBelongsTO = self.ListOfBelongingGroup(findKNeigbors)
            TestPointWithGroup = [RandomPoint,GroupPointBelongsTO]
            TestPointCheck = self.CheckIfCorrect(TestPointWithGroup)
            if TestPointCheck == True:
                timesCorrect += 1
            else: 
                timesInccorect += 1
    
        print("Amount of Time KNN sorts Correctly: ",timesCorrect, "Times KNN sorts Inccorectly: ",timesInccorect )
 

KNNTest = Classifier(10000,5,1,0,-1000,1000)
KNNTest.performKNN(6)


fileName = 'randptz.txt'
makePointsFile(1000, 2, -500, 500, fileName)
inputData = Classifier.readPointsFile(fileName)
firstClass = Classifier(inputData, 4, 1, .05)

#Testing the K-means function times
iterList = firstClass.KMeans()
print(f'Fewest number of iterations was {min(iterList)} on try {iterList.index(min(iterList)) + 1}')

randPtz = Classifier(1000, 4, 0, .05, -1000, 1000)
iterList, errorList = randPtz.KMeans()

print(f'Fewest number of iterations was {min(iterList)} on try {iterList.index(min(iterList)) + 1}\
    \nLowest error was {min(errorList)} on try {errorList.index(min(errorList)) + 1}')