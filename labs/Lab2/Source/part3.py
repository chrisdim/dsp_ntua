# Digital Signal Processing - Lab 2 - Part 3
# Christos Dimopoulos - 03117037
# Dimitris Dimos - 03117165
# PCA & k-Means Algorithm

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from numpy import load
from numpy import linalg

plt.close('all')
counter =0
# Part 3 (cont.)

energies = load('data_mxd.npy')
m,d = energies.shape

#3.2 SVD Algorithm
u, s, vh = linalg.svd(energies, full_matrices=True)
v = np.transpose(vh)

#(a) Array that holds percentage of variance for each principal component
L = np.square(s)/(m)
total_sum = np.sum(L)
L = L/total_sum

#Keep only the first two principal directions --> biggest λk
vmain = v[:,0:2]

# Principal components array Y = XV
y = np.matmul(energies,vmain)

#(b) Scatter plot the first two principal components
counter = counter+1
plt.figure(counter)
# first 20 elements step files - last 20 elements sleep files
categories1 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
colormap = np.array(['r', 'k'])
plt.scatter(y[:,0],y[:,1],c=colormap[categories1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Singular Value Decomposition - PCA")
plt.show()

#3.3 K-means Algorithm
def distance(a,b):
    one = (a[0]-b[0])**2
    two = (a[1]-b[1])**2
    return np.sqrt(one+two)

#Initialization
c1 = np.array([-59593122.7071922,-2972094.6486869])
c2 = np.array([-53971372.80773152,-3159123.49879891])
categories2 = np.zeros(40)
Distortion = 0
for i in range(40):
    Distortion = Distortion + distance(y[i,:],c1)
    
times = 0
while (True):
    # Sorting
    times = times+1
    for i in range(40):
        if distance(y[i,:],c1)<distance(y[i,:],c2):
            categories2[i] = 0
        else:
            categories2[i] = 1
        
    # Updating Centroids
    c1 = [0,0]
    c2 = [0,0]
    sum1 = 0
    sum2 = 0
    for i in range(40):
        if categories2[i]==0:
            c1 = c1 + y[i,:]
            sum1 = sum1 + 1
        else:
            c2 = c2 + y[i,:]
            sum2 = sum2 + 1
    c1 = c1/sum1
    c2 = c2/sum2
    
    # Check Termination Condition
    NewDistortion = 0
    for i in range(40):
        NewDistortion = NewDistortion + distance(y[i,:],c1)
    
    if (np.abs(NewDistortion-Distortion)<0.0000001):#threshold
        break
    else:
        Distortion = NewDistortion
        
# Scatter plot the result of k-means
counter = counter+1
plt.figure(counter)
#print(categories2)
categories2 = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1,
       1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0])
colormap = np.array(['r', 'k'])
plt.scatter(y[:,0],y[:,1],c=colormap[categories2])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("k-Means Algorithm - PCA")
plt.show()

# 3.4 Rand Index
rand_index = metrics.adjusted_rand_score(categories1,categories2)
print("Random Index = ",rand_index)