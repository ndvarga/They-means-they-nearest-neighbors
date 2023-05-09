Nikolas Varga and Wendy Quizhpi - They-Means and They-Nearest Neighbors

Introduction:
They-Means/They-Nearest Neighbors uses a class "Classifier" which has many attributes and methods. The object is initialized with the input **points** (which can be a list, array, or integer), number of clusterheads or nearest neighbors **k**, error tolerance **tolerance**, and verbosity mode **verboseMode**. There is an optional input for min and max values **minmax**, which should be used if **points** is an integer. If **points** is an integer, the class will use method **_getpoints** to create a set of points between the minimum and maximum values normally distributed about **k** centers with a variance of 1/16 the distance between the minimum and maximum values. Otherwise, it will use the data given to operate either K-means or K-nearest neighbors. General methods are a distance method **_distance**, and **readPointsFile** which will make a file with data into an array of points. Specific methods for K-means include **_formCluster** which assigns points to a clusterhead, **_updateClusterhead** which finds the mean of the points assigned to that clusterhead, and **KMeans** which runs these two functions while the distance between the new and old clusterhead locations is greater than the tolerance.

**Libraries and Tools:**
+ NumPy: NumPy is used to organize and process data. All matrices are NumPy arrays, random numbers are generated using NumPy random functions, and the NumPy mean, min, and max functions
+ matplotlib.pyplot is used to plot two-dimensional data sets visually.
+ re is used to comprehend file input.

**Lessons Learned:**

Conducting this project, we learned how to practically apply our Python knowledge to solve a (relatively) complex problem. We gained valuable experience in conceptualizing code structure, problem-solving, and using data structures efficiently to implement the KNN and K-means algorithms. We also felt that this project gave us a lot of experience in object oriented programming. We would have liked to refine the point generator and the error-checking, to see how effective each K-means attempt is. We also would have liked to spend a little bit more time on the file reader so it is more seamless. Future students should start thinking about this project early in the semester and use large blocks of time to get as much done as they can at once. For KNN, we learned that the number of KNN works is dependent on the number of groups. More groups means a greater likelihood of interspersing of different groups.

**Instructions:**
The class is provided in VargaQuizhpiFinalScript.py. You can edit this to provide different outcomes, using the Classifier object and its related attributes to run different k-means simulations

If you have a file of points:
+ Make sure they are organized in the format
```point1\n point2\n...pointn```. Points can be a list or tuple or just values separated by a space. Points must be separated by a white space character.
+ Use ```data = Classifier.readPointsFile(filename)``` then ```classifierName = Classifier(data, k, tolerance, verbosity)``` to import your data.

If you have a list or array:
+ ```classifierName = Classifier(listOrArray, k, tolerance, verbosity)```

If you want to generate random data:
+ ```classifierName = Classifier(numberOfPoints, k, tolerance, verbosity, min, max)```