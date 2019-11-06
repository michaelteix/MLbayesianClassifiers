# Author: Michael Teixeira
#     ID: 1001375188

import sys
import statistics as stat
import math

# Make sure the proper filenames have been passed to the program
assert len(sys.argv) >= 3, "Not enough command line arguments!"

class Classifier :
    def __init__(self, id) :
        self.classID = id
        self.prob = float()
        self.attributes = []

class Attribute :
    def __init__(self, attID) :
        self.attributeID = attID
        self.mean = float()
        self.stdDev = float()
        self.prob = float()
        self.values = []

class testObject :
    def __init__(self, id, trueClass) :
        self.id = id
        self.pClass = int()
        self.prob = float()
        self.tClass = trueClass
        self.accuracy = float()
        self.p_xGivenCs = []
        self.p_x = 0
    
classes = []
testObjs = []

def parceFile(file) :
    return [line.rstrip('\n') for line in open(file)]

def find_stdDev(val, mean) :
    sqrDiff = []
    
    for i in val :
        sqrDiff.append( (i - mean)**2 )

    variance = ( ( 1 / ( len(sqrDiff) - 1 ) ) * sum(sqrDiff) )
    if (variance < 0.0001) :
        variance = 0.0001

    return math.sqrt(variance)

def calc_gaussian(x, mean, stdev) :
    a = 1 / ( stdev * math.sqrt( 2 * math.pi ) )
    b = math.exp( -1 * ( (( x - mean )**2) / ( 2 * (stdev**2) ) ) )
    return a * b

def getAccuracy(classProbs, trueClass) :
    maxes = []

    for i, prob in enumerate(classProbs) :
        if(prob == max(classProbs)) :
            maxes.append(i + 1)

    if( len(maxes) == 1 ) :
        if( maxes[0] == trueClass ) :
            return 1
        else :
            return 0
    else :
        if(trueClass in maxes) :
            return 1 / len(maxes)
        else :
            return 0


def naive_bayes(training_file, test_file) :
    ### Training Phase ###
    training_data = parceFile(training_file)
    classNums = []

    # Determine # of classes
    for i in training_data :
        tempStr = i.split()
        temp = [float(x) for x in tempStr]
        
        if ( not( temp[-1] in classNums ) ) :
            classNums.append( temp[-1] )

    # Create class objects and associated attribute objects
    for i in range(0, len(classNums)) :
        classes.append(Classifier(i + 1))
        
        for j in range( 0, len( training_data[0].split() ) - 1 ) :
            classes[i].attributes.append(Attribute(j + 1))
    
    # Find and file values to associated attribute object
    for i in training_data :
        tempStr = i.split()
        temp = [float(x) for x in tempStr]
        clNum = int(temp[-1])

        for j in classes :
            if (j.classID == clNum) :
                for index, k in enumerate(temp[:-1]) :
                    j.attributes[index].values.append(k)

    # Calculate p(C)
    for k in classes :
        k.prob = len(k.attributes[0].values) / len(training_data)

    # Calulate mean and standard deviation for each attribute
    for i in classes :
        for j in i.attributes :
            if( not (len(j.values) == 0) ) :
                j.mean = stat.mean(j.values)
            j.stdDev = find_stdDev(j.values, j.mean)

    # Print results from training phase
    for i in classes :
        for j in i.attributes :
            print("Class %d, attribute %d, mean = %.2f, std = %.2f" % (i.classID, j.attributeID, j.mean, j.stdDev))
        print()
    
    ### Classification Phase ###
    test_data = parceFile(test_file)

    for index, i in enumerate(test_data) :
        tempStr = i.split()
        temp = [float(x) for x in tempStr]
        tempObj = testObject( index + 1, temp[-1] )
        
        # Calculate gaussians for each class on the test object
        for j in range( 0, len(classes) ) :
            p_xGivenC = 1

            for ind, k in enumerate(temp[:-1]) :
                tempAtt = classes[j].attributes[ind]
                p_xGivenC *= calc_gaussian( k, tempAtt.mean, tempAtt.stdDev )

            tempObj.p_xGivenCs.append(p_xGivenC)

        # Calculate p(x) with sum rule
        for j in range(0, len(classes)) :
            tempObj.p_x += ( tempObj.p_xGivenCs[j] * classes[j].prob )

        # Use Bayes Rule to calculate P(C|x)
        classProbs = []

        for j in range(0, len(classes)) :
            classProbs.append( ( tempObj.p_xGivenCs[j] * classes[j].prob ) / tempObj.p_x )

        # Take the max value as the identified class
        tempObj.prob = max(classProbs)
        tempObj.pClass = classProbs.index( max(classProbs) ) + 1
        tempObj.accuracy = getAccuracy( classProbs, tempObj.tClass )
        print("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f" % (tempObj.id, tempObj.pClass, tempObj.prob, tempObj.tClass, tempObj.accuracy))

        testObjs.append( tempObj )

    # Calculate total accuracy
    accuracySum = 0
    for i in testObjs :
        accuracySum += i.accuracy

    print("\nclassification accuracy = %6.4f" % ( accuracySum / len(testObjs) ) )


#### MAIN ####
naive_bayes(sys.argv[1], sys.argv[2])