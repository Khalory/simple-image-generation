#Featurize
from sklearn.decomposition import MiniBatchDictionaryLearning
import patches
import numpy as np

def flatten(array):
	return np.array(array).flatten()

# Returns the dictionary-based model of the provided shapes.
#Also adds the patches and components to the shape dict
def getModel(shapes, size=(9,9)):
	for shape in shapes:
		shape['patches'] = patches.toPatches(shape['img'], size)

	model = MiniBatchDictionaryLearning()
	model.fit([flatten(p) for p in shape['patches'] for shape in shapes])

	for shape in shapes:
		shape['components'] = model.transform([flatten(p) for p in shape['patches']])

	return model

def shapeFeature(name):
	return [1 if name == 'square' else 0, 1 if name == 'circle' else 0]

def featurize(shapes, size=(9,9)):
	model = getModel(shapes, size)

	features = []
	for shape in shapes:
		semiglobalFeatures = (shapeFeature(shape['shape'])
			+ [shape['x'], shape['y'], shape['size']])
		features.append([components for components in shape['components']]
			+semiglobalFeatures)

	return model, features