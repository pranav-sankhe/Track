import numpy as np
import matplotlib.pyplot as plt

# load image
img = np.array([[21.0, 20.0, 22.0, 24.0, 18.0, 11.0, 23.0],
                [21.0, 20.0, 22.0, 24.0, 18.0, 11.0, 23.0],
                [21.0, 20.0, 22.0, 24.0, 18.0, 11.0, 23.0],
                [21.0, 20.0, 22.0, 99.0, 18.0, 11.0, 23.0],
                [21.0, 20.0, 22.0, 24.0, 18.0, 11.0, 23.0],
                [21.0, 20.0, 22.0, 24.0, 18.0, 11.0, 23.0],
                [21.0, 20.0, 22.0, 24.0, 18.0, 11.0, 23.0]])
print "image =", img
grad = np.random.rand(7,7)

gradient = np.zeros(img.shape) 

width = img.shape[0]
height = img.shape[1]


for i in range(width):
	for j in range(height):
		gx, gy = np.gradient(img,img[0,:],img[:,0])		
		if np.isinf(gx[i][j]) == True:
			gx[i][j] = 0
		if np.isinf(gy[i][j]) == True:
			gy[i][j] = 0			
		gradient[i][j] = np.sqrt( np.square(gx[i][j]) + np.square(gy[i][j]) )	

print gradient