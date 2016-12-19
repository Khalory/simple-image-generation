#Shape Generator
from __future__ import division
import numpy as np
import math
import numpy.random as random

imageSize = (63,63)
minSize = 3

def blank(size=imageSize, color=0):
	return np.array([[color]*size[0]]*size[1])

def rectangle(x, y, length, width, size=imageSize, foreground=1, background=0):
	canvas = blank(size, background)
	for i in range(x, x+length+1):
		canvas[i, y] = foreground
		canvas[i, y+width] = foreground
	for j in range(y, y+width+1):
		canvas[x, j] = foreground
		canvas[x+length, j] = foreground
	return canvas


def square(x, y, length, size=imageSize, foreground=1, background=0):
	return rectangle(x, y, length, length, size, foreground, background)

def elipse(x, y, xRadius, yRadius, size=imageSize, foreground=1, background=0):
	canvas = blank(size, background)
	for i in range(size[0]):
		for j in range(size[1]):
			dist = ((i-x)/xRadius)**2 + ((j-y)/yRadius)**2-1
			if dist > -0.1 and dist < 0.1:
				canvas[i,j] = foreground

	return canvas

def circle(x, y, radius, size=imageSize, foreground=1, background=0):
	return elipse(x, y, radius, radius, size, foreground, background)

##TJ Should foreground and background get randomized as well here?
def randomSquares(samples, size=imageSize, foreground=1, background=0):
	rectangles = []
	for i in range(samples):
		length = random.randint(3, size[0])
		x = random.randint(0, size[0]-length)
		y = random.randint(0, size[1]-length)
		rect = square(x, y, length, size, foreground, background)
		rectangles.append({'shape':'square', 'img':rect,
			'x':x/size[0], 'y':y/size[0], 'size':length})
	return rectangles

def randomCircles(samples, size=imageSize, foreground=1, background=0):
	circles = []
	for i in range(samples):
		radius = random.randint(3, size[0]/2)
		x = random.randint(radius, size[0]-radius)
		y = random.randint(radius, size[1]-radius)
		circ = circle(x, y, radius, size, foreground, background)
		circles.append({'shape':'circle', 'img':circ,
			'x':x/size[0], 'y':y/size[1], 'size':radius})
	return circles


import matplotlib.pyplot as plt
import featurize
import patches
#plt.imshow(circle(15, 20, 13), cmap="Greys_r", vmin=0, vmax=1)
squares = randomCircles(50)
model = featurize.getModel(squares)
patches.showPatches(model.components_)
#plt.imshow(randomCircles(1)[0]['img'], cmap="Greys_r", vmin=0, vmax=1)
#plt.show()
circ = circle(15, 20, 11)
#plt.figure(99)
#plt.imshow(circ, cmap='Greys_r', vmin=0, vmax=1)
#patches.showPatches(patches.toDensePatches(circ))