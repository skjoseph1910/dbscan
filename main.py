from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import random
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pyclustering.cluster.kmedoids as km
from sklearn.metrics import silhouette_score
from sklearn import manifold

def students(array):
    dict = {}
    another = {}
    for num, i in enumerate(array):
        if i not in dict:
            dict[i] = []
            another[i] = []
        dict[i].append('student ' + str(num + 1))
        another[i].append(num)
    
    return dict, another

def calcualtecohesion(dict, array):
    final = []
    for i, j in dict.items():
        list = []
        for p in j:
            list.append(array[p])
        x = 0
        y = 0
        for a in list:
            x += a[0]
            y+= a[1]
        
        x = x/len(list)
        y= y/len(list)
        if x < 0 :
            x = x*-1
        if y < 0 :
            y = y*-1
        
        cohesion = 0
        for l in list:
            if l[0] <0:
                l[0] = l[0]*-1
            if l[1] <0:
                l[1] = l[1]*-1
            cohesion += (l[0]-x)*(l[0]-x)/10
            cohesion += (l[1]-y)*(l[1]-y)/10
        final.append(cohesion)

    return final

df = pd.read_excel('hello.xlsx')
array=np.array(df)

#distance matrix
DM = pairwise_distances(array, metric = 'euclidean')


clustering = DBSCAN(eps=25, min_samples=2, metric = 'euclidean').fit(DM)

y= clustering.fit_predict(DM)
avg = silhouette_score(DM, y)

#distance matrix to x,y coordinates to calculate cohesion
mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state = 3)
result = mds.fit(DM)
coords = result.embedding_
#print(type(coords))
#clusters with students

dict, another = students(y)
cohesion = calcualtecohesion(another,coords)

count = 0
for i,j in dict.items():
    print('cluster:', j, 'cohesion:', cohesion[count])
    count +=1
print('silhouette', avg)

#to see x,y plot
'''
count =0
for i, j in another.items():
    x= []
    y = []
    colors = ['blue', 'green', 'red', 'cyan', 'yellow', 'black', 'magenta']
    for p in j:
        x.append(coords[p][0])
        y.append(coords[p][1])
    plt.scatter(x, y, c = colors[count])
    count +=1
plt.show()
'''

