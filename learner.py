# LSTM for international airline passengers problem with memory
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error

#Theano flags
#import theano
#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'

##TJ
import shapeGenerator
import featurize
import patcher

look_back = 20
batch_size = 3005 # number of patches per image ##TJ

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	fullX, fullY = [], []
	print('Creating dataset')
	for shape in dataset:
		dataX, dataY = [], []
		for i in range(len(shape)-look_back):
			dataX.append(shape[i:(i+look_back), :])
			dataY.append(shape[i + look_back, :])
		fullX.append(dataX)
		fullY.append(dataY)
	return np.array(fullX), np.array(fullY)

# fix random seed for reproducibility
np.random.seed(7)

dataset = shapeGenerator.randomSquares(25) + shapeGenerator.randomCircles(25)
dimredModel, features = featurize.featurize(dataset)
features = np.array(features)

# split into train and test sets
train_size = int(round(len(dataset) * 0.67))
test_size = len(dataset) - train_size
print('Train {0}, test {1}'.format(train_size, test_size))
train, test = features[0:train_size,:], features[train_size:len(features),:]

# reshape into X=t and Y=t+1
print('Z')
print(train.shape)
trainX, trainY = create_dataset(train, look_back)
print(trainX.shape)
print(trainY.shape)
testX, testY = create_dataset(test, look_back)

print('F')

# create and fit the LSTM network
model = Sequential()
##TJ Expected at least 2 hidden layers but preliminary testing to be done with 1
#model.add(LSTM(100, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
model.add(LSTM(50, batch_input_shape=(1, look_back, featurize.numFeatures)))
model.add(Dense(featurize.numFeatures))
model.compile(loss='mean_squared_error', optimizer='adam')

print('G')

iterations = 2
for i in range(iterations):
	for j, x in enumerate(trainX):
		y = trainY[j]
		for k in range(len(x)):
			model.fit(np.reshape(x[k], (1, len(x[k]), len(x[k][0]))), np.reshape(y[k], (1, len(y[k]))), nb_epoch=1, batch_size=1, verbose=0, shuffle=False)
		model.reset_states()

print('H')

# make predictions
trainPredict = trainX[0][0]
trainScore = 0
for i, x in enumerate(trainX[0]):
	nextTrain = model.predict(np.reshape(trainPredict, (1, len(trainPredict), len(trainPredict[0]))), batch_size=batch_size)
	trainPredict[0:look_back-2] = trainPredict[1:look_back-1]
	trainPredict[look_back-1] = nextTrain
	trainScore += math.sqrt(mean_squared_error(np.reshape(trainY[0][i], (1, len(trainY[0][i]))), nextTrain))
print('Train Score: %.2f RMSE' % (trainScore))

model.reset_states()

testPredict = testX[0][0]
testScore = 0
for i, x in enumerate(testX[0]):
	nextTest = model.predict(np.reshape(testPredict, (1, len(testPredict), len(testPredict[0]))), batch_size=batch_size)
	testPredict[0:look_back-2] = testPredict[1:look_back-1]
	testPredict[look_back-1] = nextTest
	testScore += math.sqrt(mean_squared_error(np.reshape(testY[0][i], (1, len(testY[0][i]))), nextTest))
	
model.reset_states()

# calculate root mean squared error
print('Test Score: %.2f RMSE' % (testScore))

# Testing Full Image Generation
for imgIndex in range(len(testX[0])):
	predict = testX[0][imgIndex]
	print(np.array(predict).shape)
	allPatches = predict[0:look_back-1]
	for i in range(batch_size+1):
		nextPatch = model.predict(np.reshape(predict, (1, len(predict), len(predict[0]))), batch_size=1)
		predict[0:look_back-2] = predict[1:look_back-1]
		predict[look_back-1] = nextPatch
		allPatches = np.vstack((allPatches, nextPatch))

	model.reset_states()
	allPatches = allPatches[:,0:featurize.numComponents]
	reconstructedPatches = featurize.reconstructPatches(dimredModel, allPatches)
	finalImg = patcher.patchesToImg(reconstructedPatches)
	print(np.array(finalImg).shape)
	try:
		print(mean_squared_error(reconstructedPatches, dataset[train_size+imgIndex]['patches']))
	except Exception as ex:
		print(ex)

	plt.figure(figsize=(4.2, 4))
	plt.subplot(2, 1, 1)
	plt.imshow(dataset[train_size+imgIndex]['img'], cmap='Greys_r')
	plt.subplot(2, 1, 2)
	plt.imshow(finalImg, cmap='Greys_r', vmin=0)
	plt.show()

# shift train predictions for plotting
#trainPredictPlot = np.empty_like(dataset)
#trainPredictPlot[:, :] = np.nan
#trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
## shift test predictions for plotting
#testPredictPlot = np.empty_like(dataset)
#testPredictPlot[:, :] = np.nan
#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
## plot baseline and predictions
#plt.plot(scaler.inverse_transform(dataset))
#plt.plot(trainPredictPlot)
#plt.plot(testPredictPlot)
#plt.show()
