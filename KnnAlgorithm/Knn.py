import numpy as np
import operator as op

# import numpy downloads the numpy package and links it to the local variable 
# of my choice that is np . Only import numpy will lead to a line equivalent to 
# import numpy as numpy
# import numpy as np is programmatically equivalent as:-
# import numpy as numpy
# np = numpy
#del numpy
# import submodules of your choice  

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    # use the local variable that stores the modules to call a function
    labels = ['A','A','B','B']
    return group,labels

# This is the implementation of kNN algorithm