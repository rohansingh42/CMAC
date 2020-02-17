#!/usr/bin/env python

'''
ENPM 690 Spring 2020: Robot Learning
Homework 2: CMAC(1D Discrete)

Author:
Rohan Singh (rohan42@terpmail.umd.edu)
Graduate Student in Robotics,
University of Maryland, College Park
'''
import matplotlib.pyplot as plt
import numpy as np

nDataPoints = 100
trainingDataSize = 70
testingDataSize = 30
nWeights = 35
g = 5
learningRate = 0.1
testing_error = []
nEpochs = 100

dataRange = 10
x = np.arange(0,dataRange,0.1)   # start,stop,step
y = np.sin(x)

trainingIndices = np.random.choice(np.arange(nDataPoints), size = 70, replace = False).tolist()
trainingIndices.sort()
trainingData = [[x[index], y[index]] for index in trainingIndices]

testingIndices = []

for i in np.arange(100):
	if i not in trainingIndices:
		testingIndices.append(i)

testingData = [[x[index], y[index]] for index in testingIndices]

# Initialize Weights for Mapping
weightArr = np.ones(35).tolist()

perEpochData = []

for episode in range(nEpochs):
	errors = []
	predictedOutputs = []

	for index in range(trainingDataSize):
		ip = trainingData[index][0]
		desOutput = trainingData[index][1]

		# Find association window upper and lower limits for output calculation
		associationCenter = nWeights*(ip/dataRange)
		if associationCenter - g/2 < 0:
			lower = 0
		else :
			lower = associationCenter - g/2

		if associationCenter + g/2 > (nWeights-1):
			upper = nWeights-1
		else:
			upper = associationCenter + g/2

		# Calculate output from Network
		predOutput = 0
		for i in range(int(lower)+1, int(upper)):
			predOutput = predOutput + weightArr[i]*ip
		predOutput = predOutput + weightArr[int(lower)]*ip*(int(lower)+1-lower)
		predOutput = predOutput + weightArr[int(upper)]*ip*(upper - int(upper))

		predictedOutputs.append(predOutput)

		# Calculate Error
		error = desOutput - predOutput
		errors.append(error)

		# Update weights
		for i in range(int(lower)+1, int(upper)):
			weightArr[i] = weightArr[i] + learningRate*error/(upper - lower)
		weightArr[int(lower)] = weightArr[int(lower)] + learningRate*error*(int(lower)+1-lower)/(upper - lower)
		weightArr[int(upper)] = weightArr[int(upper)] + learningRate*error*(upper - int(upper))/(upper - lower)

	perEpochData.append([predictedOutputs, errors, weightArr])

errors = []
predictedOutputs = []
for index in range(testingDataSize):
	ip = testingData[index][0]
	desOutput = testingData[index][1]

	# Find association window upper and lower limits for output calculation
	associationCenter = nWeights*(ip/dataRange)
	if associationCenter - g/2 < 0:
		lower = 0
	else :
		lower = associationCenter - g/2

	if associationCenter + g/2 > (nWeights-1):
		upper = nWeights-1
	else:
		upper = associationCenter + g/2

	# Calculate output from Network
	predOutput = 0
	for i in range(int(lower)+1, int(upper)):
		predOutput = predOutput + weightArr[i]*ip
	predOutput = predOutput + weightArr[int(lower)]*ip*(int(lower)+1-lower)
	predOutput = predOutput + weightArr[int(upper)]*ip*(upper - int(upper))

	print((int(lower)+1-lower), (upper - int(upper)))
	predictedOutputs.append(predOutput)

	# Calculate Error
	error = desOutput - predOutput
	errors.append(error)

# plt.plot(x,y)
# plt.plot([testingData[index][0] for index in range(testingDataSize)], predictedOutputs)

plt.plot(range(nEpochs), [sum(perEpochData[index][1]) for index in range(nEpochs)])

mse = 0
for i in range(testingDataSize):
	mse += np.linalg.norm(y[testingIndices[i]] - predictedOutputs[i])

mse = mse/np.sqrt(testingDataSize)

print(mse)

plt.show()