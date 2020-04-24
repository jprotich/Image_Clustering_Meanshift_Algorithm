import sys
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from PIL import Image
from time import process_time

fileInput = input("Enter an file name located in this directory, along with its extension: ")
time_start = process_time()

try:
    image = Image.open(fileInput)
except FileNotFoundError:
    print('Oops! File does not exist in this directory.')
    sys.exit()

imageArr = np.array(image)
imageDim = imageArr.shape
print("Size of image: {}x{}".format(imageDim[0], imageDim[1]))

#Image becomes 2D array; colors will be altered
imageArr = np.reshape(imageArr, [-1,3])

#Estimate bandwidth
bandwidth = estimate_bandwidth(imageArr, quantile=0.1, n_samples=1000)    
print("Bandwidth: ", bandwidth)

#Run Mean shift algorithm
ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)    
ms.fit(imageArr)

#Cluster labels (represented by RGB) and cluster centers    
clusterLabels=ms.labels_   
clusterCenters = ms.cluster_centers_       
clusterNum = len(np.unique(clusterLabels))    
print("Number of clusters: ", clusterNum)    

#Display 3D scatter plot of cluster groupings for segmented image
fig = plt.figure(1)
ax = fig.add_subplot(111, projection = '3d')
scatter = ax.scatter(imageArr[:,0], imageArr[:,1], imageArr[:,2], marker='o', c = clusterLabels)
ax.scatter(clusterCenters[:,0], clusterCenters[:,1], clusterCenters[:,2], marker='+', color='red', s =  100, linewidth=2, zorder=10)
ax.set_xlim3d(0,255)
ax.set_ylim3d(0,255)
ax.set_zlim3d(0,255)
plt.show()

#Display comparison of original vs. clustered image
segmentedImg = np.reshape(clusterLabels, imageDim[:2])

plt.figure(2)
plt.subplot(121)
plt.imshow(image)
plt.axis('off')

plt.subplot(122)
plt.imshow(segmentedImg)
plt.axis('off')
plt.show()

time_stop =  process_time()
print("Processing time: ", time_stop)
