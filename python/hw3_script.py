import os
import csv
import numpy as np
import NB

# Point to data directory here
# By default, we are pointing to '../data/'
data_dir = os.path.join('..','data')

# Read vocabulary into a list
# You will not need the vocabulary for any of the homework questions.
# It is provided for your reference.
with open(os.path.join(data_dir, 'vocabulary.csv'), 'rt') as f:
    reader = csv.reader(f)
    vocabulary = list(x[0] for x in reader)

#print(len(vocabulary))

# Load numeric data files into numpy arrays
XTrain = np.genfromtxt(os.path.join(data_dir, 'XTrain.csv'), delimiter=',')
yTrain = np.genfromtxt(os.path.join(data_dir, 'yTrain.csv'), delimiter=',')
XTrainSmall = np.genfromtxt(os.path.join(data_dir, 'XTrainsmall.csv'), delimiter=',')
yTrainSmall = np.genfromtxt(os.path.join(data_dir, 'yTrainsmall.csv'), delimiter=',')
XTest = np.genfromtxt(os.path.join(data_dir, 'XTest.csv'), delimiter=',')
yTest = np.genfromtxt(os.path.join(data_dir, 'yTest.csv'), delimiter=',')

# TODO: Test logProd function, defined in NB.py
''' 

# TODO: Test NB_XGivenY function, defined in NB.py
beta_0=5
beta_1=7
D1=NB.NB_XGivenY(XTrain, yTrain, beta_0, beta_1)
D2=NB.NB_XGivenY(XTrainSmall, yTrainSmall, beta_0, beta_1)


# TODO: Test NB_YPrior function, defined in NB.py
p1=NB.NB_YPrior(yTrain)
p2=NB.NB_YPrior(yTrainSmall)


# TODO: Test NB_Classify function, defined in NB.py
yHat1=NB.NB_Classify(D1, p1, XTrain)
yHat2=NB.NB_Classify(D2, p2, XTrainSmall)
print(NB.classificationError(yHat1, yTrain))
print(NB.classificationError(yHat2, yTrainSmall))
'''

# TODO: Test classificationError function, defined in NB.py

# TODO: Run experiments outlined in HW2 PDF
D1=NB.NB_XGivenY(XTrain, yTrain, 5, 7)
D2=NB.NB_XGivenY(XTrain, yTrain, 7, 5)


# TODO: Test NB_YPrior function, defined in NB.py
p1=NB.NB_YPrior(yTrain)



# TODO: Test NB_Classify function, defined in NB.py
yHat1=NB.NB_Classify(D1, p1, XTrain)
yHat2=NB.NB_Classify(D2, p1, XTrain)
print(NB.classificationError(yHat1, yTrain))
print(NB.classificationError(yHat2, yTrain))