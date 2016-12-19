from sklearn.feature_extraction import image
import numpy as np
import matplotlib.pyplot as plt
import math


def toPatches(img, shape=(9,9)):
	return image.extract_patches_2d(img, shape)

def toDensePatches(img, shape=(9,9)):
	dx, dy = shape
	patcheses = [[0]*shape[1]]*shape[0]
	for i in range(dx):
		for j in range(dy):
			patcheses[i][j] = image.extract_patches_2d(img[i:i-dx][j:j-dy], shape)
	return np.array(patcheses).reshape(-1,9,9)

def showPatches(patches, shape=(9,9)):
	plt.figure(figsize=(4.2, 4))
	size = 9
	for i, comp in enumerate(patches[:min(patches.shape[0], 400)]):
		plt.subplot(size, size, i + 1)
		plt.imshow(comp.reshape(shape), cmap='Greys_r', vmin=0, vmax=1)
		plt.xticks(())
		plt.yticks(())
	#plt.suptitle('Dictionary learned from face patches\n' +
	#	'Train time %.1fs on %d patches' % (dt, len(data)),
	#	fontsize=16)
	plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
	plt.show()
