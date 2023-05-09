import numpy as np
import matplotlib.pyplot as plt
import re

# Function to make a file with nPoints, nDims, between minVal and maxVal.
# Chooses a random number of centers and creates points normally distributed around centers.
def makePointsFile(nPoints, nDims, minVal, maxVal, filename):
        middleVal = np.mean([abs(maxVal), abs(minVal)])
        nCenters = np.random.randint(1,10)
        evenDiv = int(np.floor(nPoints/nCenters))
        extraPoints = nPoints-(nCenters*evenDiv)
        ptz = np.zeros((nDims,nPoints))
        for k in range(nCenters):
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
    def __init__(self, points, numClusterheads, tolerance, verboseMode, *minmaxVal): #add input for data set, take vectors from random gaussian distribution and compare
        if minmaxVal and isinstance(points, int):
            self.__maxVal = max(minmaxVal)
            self.__minVal = min(minmaxVal)
        elif isinstance(points, int):
            raise ValueError('Must have min and max values to generate points')
        else:
            self.__maxVal = np.max(points)
            self.__minVal = np.min(points)
        self.k = numClusterheads
        self.verbosity = verboseMode
        self.n = points
        self._center = self._points.mean()
        self.tolerance = tolerance

    @property #getter for numPoints
    def n(self):
        return self._n

    @n.setter #setter for _points and _n - must be positive int or list
    def n(self, newPoints):
        if isinstance(newPoints, int) and newPoints > 0:
            self._n = newPoints
            self._numDims = int(input('How many dimensions?'))
            self._points = self._getPoints(newPoints)
        elif isinstance(newPoints, list):
            self._points = np.array(newPoints)
            self._points = newPoints
            nDim, x = self._points.shape
            if x < nDim:
                self._points.T
            self._n = max(x,nDim)
            self._numDims = min(x,nDim)
        elif isinstance(newPoints, np.ndarray):
            self._points = newPoints
            nDim, x = self._points.shape
            if x < nDim:
                self._points.T
            self._n = max(x,nDim)
            self._numDims = min(x,nDim) 
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
        middleVal = np.mean([abs(self.__maxVal), abs(self.__minVal)])
        evenDiv = int(np.floor(self.n/self.k))
        extraPoints = self.n-(self.k*evenDiv)
        pointz = np.zeros((self._numDims,self.n))
        for k in range(self.k):
            center = np.random.randint(np.floor(2*self.__minVal/3),np.floor(2*self.__maxVal/3))
            trifecta = np.array([[np.random.normal(center, middleVal/8) for j in range(evenDiv)] for n in range(self._numDims)])
            pointz[:,k*evenDiv:(k+1)*evenDiv] = trifecta
        if extraPoints:
            pointz[:,-extraPoints:]=[[np.random.normal(0, middleVal/8) for i in range(extraPoints)] for n in range(self._numDims)]
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
                '''if np.isnan(self.clusterheads[0,j]):
                    self._randClusterhead(i) '''
                pointDist = np.append(pointDist, self._distance(self._points[:,i], self.clusterheads[:,j]))
            for ind in np.where(pointDist == pointDist.min()):
                if ind.any():
                    self.assignmentMat[0, i] = ind[0]
            
            distMat[i,:] = pointDist
        
        self.__distances = distMat
        # print(f'distances:\n{distMat}')          #maybe should use this instead of assignmentMat


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
                    #print(f' {match.group(1)} | {match.group(2)}')
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
        plt.show()
    
    def KMeans(self):
        self._firstClusterhead()
        self._formCluster()
        self._updateClusterhead()
        numIter = 0
        while self._distance(self.clusterheads, self.lastClusterheads) > self.tolerance:
            print(f'distance between old and new heads: {self._distance(self.clusterheads, self.lastClusterheads):.2f}')
            self._formCluster()
            self._updateClusterhead()
            numIter+=1
        print(f'Finished try! Distance between old and new heads: {self._distance(self.clusterheads, self.lastClusterheads):f}\n')
        if self.verbosity == 1 and self._numDims == 2:
            self.seePlot()
        return numIter     

fileName = 'randptz.txt'
makePointsFile(1000, 2, -500, 500, fileName)
inputData = Classifier.readPointsFile(fileName)

firstClass = Classifier(inputData, 5, .05, 1)
errorList = []
#Testing the K-means function 10 times
for i in range(10):
    numIter = firstClass.KMeans()
    errorList.append(numIter)
print(f'Fewest number of iterations was {min(errorList)} on try {errorList.index(min(errorList)) + 1}')

