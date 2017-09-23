import math
import numpy as np

# The logProd function takes a vector of numbers in logspace 
# (i.e., x[i] = log p[i]) and returns the product of those numbers in logspace
# (i.e., logProd(x) = log(product_i p[i]))
def logProd(x):
	## Inputs ## 
	# x - 1D numpy ndarray
	
	## Outputs ##
	# log_product - float
	log_product = sum(x)
	return log_product

# The NB_XGivenY function takes a training set XTrain and yTrain and
# Beta parameters beta_0 and beta_1, then returns a matrix containing
# MAP estimates of theta_yw for all words w and class labels y
def NB_XGivenY(XTrain, yTrain, beta_0, beta_1):
	## Inputs ## 
	# XTrain - (n by V) numpy ndarray
	# yTrain - 1D numpy ndarray of length n
	# alpha - float
	# beta - float
	
	## Outputs ##
	# D - (2 by V) numpy ndarray	

	D = np.zeros([2, XTrain.shape[1]])

	for j in range(XTrain.shape[1]):
		alpha_0_pos=0
		alpha_0_neg=0
		alpha_1_pos=0
		alpha_1_neg=0
		for i in range(XTrain.shape[0]):
			if (XTrain[i][j]==1 and yTrain[i]==1):
				alpha_0_pos+=1
			elif (XTrain[i][j]==0 and yTrain[i]==1):
				alpha_1_pos+=1
			elif (XTrain[i][j]==1 and yTrain[i]==0):
				alpha_0_neg+=1
			elif (XTrain[i][j]==0 and yTrain[i]==0):
				alpha_1_neg+=1

		D[0][j]=float(alpha_0_neg+beta_0-1)/float(alpha_0_neg+beta_0-1 + alpha_1_neg+beta_1-1)
		D[1][j]=float(alpha_0_pos+beta_0-1)/float(alpha_0_pos+beta_0-1 + alpha_1_pos+beta_1-1)

		#print("alpha_0_neg ="+ str(alpha_0_neg) +" alpha_1_neg=" + str( alpha_1_neg) +" alpha_0_pos ="+ str(alpha_0_pos) +" alpha_1_pos=" + str(alpha_1_pos))

	
	return D
	
# The NB_YPrior function takes a set of training labels yTrain and
# returns the prior probability for class label 0
def NB_YPrior(yTrain):
	## Inputs ## 
	# yTrain - 1D numpy ndarray of length n

	## Outputs ##
	# p - float

	p = 1-np.mean(yTrain)
	return p

# The NB_Classify function takes a matrix of MAP estimates for theta_yw,
# the prior probability for class 0, and uses these estimates to classify
# a test set.
def NB_Classify(D, p, XTest):
	## Inputs ## 
	# D - (2 by V) numpy ndarray
	# p - float
	# XTest - (m by V) numpy ndarray
	
	## Outputs ##
	# yHat - 1D numpy ndarray of length m

	yHat = np.ones(XTest.shape[0])
	for i in range(XTest.shape[0]):
		vector_pos=[]
		vector_neg=[]
		vector_pos.append(math.log(1-p))
		vector_neg.append(math.log(p))
		for j in range(XTest.shape[1]):
			if XTest[i][j]==1:
				vector_pos.append(math.log(D[1][j]))
				vector_neg.append(math.log(D[0][j]))
			else:
				vector_pos.append(math.log(1-D[1][j]))
				vector_neg.append(math.log(1-D[0][j]))

				
		if (logProd(vector_neg)>=logProd(vector_pos)):
			yHat[i]=0
		else:
			yHat[i]=1

	return yHat

# The classificationError function takes two 1D arrays of class labels
# and returns the proportion of entries that disagree
def classificationError(yHat, yTruth):
	## Inputs ## 
	# yHat - 1D numpy ndarray of length m
	# yTruth - 1D numpy ndarray of length m
	
	## Outputs ##
	# error - float

	error = 0
	for i in range(yTruth.shape[0]):
		if yTruth[i]!=yHat[i]:
			error+=1

	error=float(error)/float(yTruth.shape[0])
	return error
