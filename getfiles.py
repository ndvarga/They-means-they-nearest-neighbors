import numpy as np
import re


def makePointsFile(nPoints, nDims, minVal, maxVal):
    middleVal = np.mean([abs(maxVal), abs(minVal)])
    nCenters = np.random.randint(1,10)
    evenDiv = int(np.floor(nPoints/nCenters))
    extraPoints = nPoints-(nCenters*evenDiv)
    ptz = np.zeros((nDims,nPoints))
    for k in range(nCenters):
       center = np.random.randint(np.floor(2*minVal/3),np.floor(2*maxVal/3))
       stupidfuckingList = [[np.random.normal(center, middleVal/8) for j in range(evenDiv)] for n in range(nDims)]
       trifecta = np.array(stupidfuckingList)
       ptz[:,k*evenDiv:(k+1)*evenDiv] = trifecta
       if extraPoints:
           ptz[:,-extraPoints:]=[[np.random.normal(center, middleVal/8) for i in range(extraPoints)] for n in range(nDims)]
    print(ptz)
    ptzUse = ptz.T
  
    ptz2write = np.array2string(ptz.T)
  
    with open('randptz.txt', 'w') as curPoints:
        for point in ptzUse:
           ptz2write = np.array2string(point) + '\n'
           curPoints.write(ptz2write)
 
makePointsFile(10,2,-100,100)
 
def readPointsFile(filename):
   if isinstance(filename, str):
       with open(filename, 'r') as readPoints:
           pointsList = readPoints.read()
           curPattern = re.compile(r'\[(.{6,12})\s{1,3}(.{6,12})\]', flags=re.MULTILINE)
           curMatch = curPattern.finditer(pointsList)
           dataIn = []
           for match in curMatch:
               print(f' {match.group(1)} | {match.group(2)}')
               newPt = [float(match.group(1)), float(match.group(2))]
               dataIn.append(newPt)
           print(f'comprehended version: \n{dataIn}')
           dataIn = np.array(dataIn).T
           print(f'regular version: \n{dataIn}')
           return dataIn
   else:
       raise ValueError('File name must be a string.')
  
readPointsFile('randptz.txt')