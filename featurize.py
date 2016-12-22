#Featurize
#from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
import patcher
import numpy as np

numGlobalFeatures = 5
numComponents = 40
numFeatures = numComponents + numGlobalFeatures

def flattenFirst(arr):
	flat = np.zeros((len(arr[:])*len(arr[0][:]), len(arr[0][0][:])))
	for i, x in enumerate(arr):
		for j, y in enumerate(x):
			#print(y)
			flat[i*len(x)+j] = flatten(y)
	return flat

def flattenLast(arr):
	flat = np.zeros((len(arr[:]), len(arr[0][:]), len(arr[0][0][:])*len(arr[0][0][0][:])))
	for i, x in enumerate(arr):
		for j, y in enumerate(x):
			#print(y)
			flat[i][j] = flatten(y)
	return flat

def flatten(array):
	return np.array(array).flatten()

def reconstructPatches(model, features):
	return [np.dot(x, model.components_) for x in features]

# Returns the dictionary-based model of the provided shapes.
#Also adds the patches and components to the shape dict
def getModel(shapes, size=(9,9)):
	for shape in shapes:
		shape['patches'] = patcher.toDensePatches(shape['img'], size)

#	model = FastICA(n_components = numComponents)
	model = PCA(n_components = numComponents)
	patches = [shape['patches'] for shape in shapes]
	print('Patches shape: {0}'.format(np.array(patches).shape))
	flat = flattenLast(patches)
	print('Flattened shape: {0}'.format(flat.shape))
	model.fit(flattenFirst(flat))
	
	for i, shape in enumerate(shapes):
		shape['components'] = model.transform(flat[i])

	return model

def shapeFeature(name):
	return [1 if name == 'square' else 0, 1 if name == 'circle' else 0]

def featurize(shapes, size=(9,9)):
	model = getModel(shapes, size)
	x, y = size[0], size[1]
	features = np.zeros((len(shapes), len(shapes[0]['patches']), numFeatures))
	for i, shape in enumerate(shapes):
		semiglobalFeatures = (shapeFeature(shape['shape']) +
			[shape['x'], shape['y'], shape['size']])
		shapeComponents = np.concatenate((shape['components'],
			([semiglobalFeatures]*len(shape['components']))), axis=1)
		features[i] = shapeComponents

	return model, features

import shapeGenerator as sg
from sklearn.metrics import mean_squared_error
circs = sg.randomCircles(1)
patches = flattenLast([patcher.toDensePatches(circs[0]['img'])])
dimredModel, features = featurize(circs)
reconstructedPatches = reconstructPatches(dimredModel, np.reshape(features[:,:,0:numComponents], (-1, numComponents)))
print('MSE2')
patches = np.reshape(patches, (len(patches[0]), -1))
print(np.array(patches).shape)
print(np.array(reconstructedPatches).shape)
print(mean_squared_error(patches, reconstructedPatches))

finalImg = patcher.patchesToImg(reconstructedPatches)
print('MSE3')
print(mean_squared_error(circs[0]['img'], finalImg))

import matplotlib.pyplot as plt
plt.figure(figsize=(4.2, 4))
plt.subplot(1, 2, 1)
plt.imshow(circs[0]['img'], cmap='Greys_r', vmin=0, vmax=1)
plt.subplot(1, 2, 2)
plt.imshow(finalImg, cmap='Greys_r', vmin=0)
plt.show()

