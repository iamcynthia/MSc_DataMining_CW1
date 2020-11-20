#### 2. Clustering ####

#### initial setting ####
import numpy as np
import pandas as pd
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
#### Loading Data ####
sale = pd.read_csv("wholesale_customers.csv")
sale = sale.drop(['Channel', 'Region'], axis =1)


#### Q2-1 ####
ran = []
m = []
for i in range(len(sale.iloc[0,:])):
    u = (sale.iloc[:,i].values.sum()) / len(sale.iloc[:,0])  # compute the mean
    m.append(u)
    r = [min(sale.iloc[:,i]) , max(sale.iloc[:,i])]  # compute the range
    ran.append(r)
table = pd.DataFrame({'mean' : m, 'range' : ran}, index=['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents', 'Delicatessen'])
print 'Question 2-1 : '
print table


#### Q2-2 ####
## combine the data in each pair of attributes to list k
k = []
for i in range(5):
    k.append(sale.iloc[:,0:i+2:i+1].values.tolist())
for i in range(4):
    k.append(sale.iloc[:,1:i+3:i+1].values.tolist())
for i in range(3):
    k.append(sale.iloc[:,2:i+4:i+1].values.tolist())
for i in range(2):
    k.append(sale.iloc[:,3:i+5:i+1].values.tolist())
for i in range(1):
    k.append(sale.iloc[:,4:i+6:i+1].values.tolist())
k = np.asarray(k)

## create a list of title (the pair of attributes)
q = 5
w = 2
l = [0,1,2,3,4]
title = []
label = []
for j in range(5):
    for s in range(w,7):
        for i in range(q):
            title.append( "%s & %s"%(sale.iloc[:,j:i+s:i+1].keys()[0] , sale.iloc[:,j:i+s:i+1].keys()[1])  )
            label.append(sale.iloc[:,j:i+s:i+1].keys()[0])
            label.append(sale.iloc[:,j:i+s:i+1].keys()[1])
        j = j +1 
        q = q-1
    w = w+1 

## 15 scatter plots
xlabel = [label[j] for j in range(0,30,2)]
ylabel = [label[s] for s in range(1,30,2)]
km = cluster.KMeans(n_clusters = 3, random_state = 0)
fig=plt.figure(figsize=(20, 25))
for i in range(15):
    km.fit(sale)
    plt.subplot(5,3,i+1)
    plt.scatter(k[i][:,0],k[i][:,1], c=km.labels_)
    plt.title('KMeans Scatterplot (k = 3)')
    plt.xlabel(xlabel[i])
    plt.ylabel(ylabel[i])
plt.show()


## Q2-3 ##
# this function is about calculating the distance between two vectors
def dist(vec1, vec2):
    return np.sum(np.square(vec1-vec2))

# this function is to calculate between cluster distance
def BC(n):
    b = np.zeros(n)
    km = cluster.KMeans(n_clusters = n, random_state = 0)
    km.fit(sale)
    l = 1
    arr = []
    for j in range(n-1):
        for i in range(l,n):
            arr.append(dist(km.cluster_centers_[j] ,km.cluster_centers_[i]))
        l+=1
    arr = np.asarray(arr)
    return arr.sum()

# this function is to calculate within cluster distance
def WC(n):
    km = cluster.KMeans(n_clusters = n, random_state = 0)
    km.fit(sale)
    arr = []
    for h in range(len(sale)):
        for i in range(n):
            if km.labels_[h] == i: 
                arr.append(dist(sale.iloc[h,:],km.cluster_centers_[i]))
    arr = np.asarray(arr)
    return arr.sum()

# this function is to calculate the ration BC/WC
def ratio(n):
    rate = '%.2f%%' % (BC(n)/WC(n) * 100)
    return rate

table = pd.DataFrame({'k = 3' : [int(BC(3)), int(WC(3)), ratio(3)], 'k = 5' : [int(BC(5)), int(WC(5)), ratio(5)], 
                    'k = 10' : [int(BC(10)), int(WC(10)), ratio(10)]}, index = ['BC', 'WC', 'BC/WC'], columns = ['k = 3', 'k = 5', 'k = 10'])
print 'Question 2-3 : '
print table