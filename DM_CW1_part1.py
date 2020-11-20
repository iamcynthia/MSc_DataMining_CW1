#### 1. Classification ####

#### initial setting ####
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
#### Loading Data ####
adult = pd.read_csv("adult.csv")
adult = adult.drop(['fnlwgt'], axis =1) 


## Q1-1 ##
# (i) number of instances
num_ins = len(adult)  

# (ii) num of missing values
na = adult.isnull().sum().sum()
na = na.astype('float64') 

# (iii) fraction of missing values over all attribute values
all_attr = adult.count().sum()  # all attribute values
p_na_attr = round(na/all_attr,4)
p_na_attr = '%.2f%%' % (p_na_attr * 100)

# (iv) num of instances with missing values
na_ins = sum([True for idx,row in adult.iterrows() if any(row.isnull())])  

# (v) fraction of instances with missing values over all instances
p_na_ins = round(na_ins/float(num_ins), 4)
p_na_ins = '%.2f%%' % (p_na_ins * 100)

# create a list about the information from question
information = ["number of instances", "number of missing value", "fraction of missing values over all attribute values", 
               "number of instances with missing values", "fraction of instances with missing values over all instances"]
# a list of the answer of the above information
counts = [num_ins, na, p_na_attr, na_ins, p_na_ins] 

df = pd.DataFrame(zip(information, counts), columns = ["information", "counts"])
print 'Question 1-1 : '
print df


## Q1 - 2
without_na = adult.dropna()  # delete the instances which has NaN
nom = without_na.apply(LabelEncoder().fit_transform) # dataframe which has transformed into nominal

print 'Question 1-2 : '
for i in range(len(nom.iloc[0,:13])):
    print "%s : "%(nom.keys()[i]),sorted(nom.iloc[:,i].unique()) # print the attributes of the dataframe nom and the unique label for each attribute


## Q1-3 ##
## the former 12 attrbutes' vector represents 'data'
data = [nom.iloc[i,0:13].tolist() for i in range(len(nom))]
## the last attribute represents 'target'
target = nom.iloc[:,13].tolist()

## create decision tree classifier and train
train_data, test_data, train_label, test_label = train_test_split(data, target, test_size=0.2, random_state=0)
tree = tree.DecisionTreeClassifier(random_state=0)
tree.fit(train_data, train_label)

print 'Question 1-3 : '
## calculate the error rate 
error = 0
for i, v in enumerate(tree.predict(test_data)):
    if v != test_label[i]:  # if the predicted data isn't same as test label, then add error once
        error+=1
e_rate = error/float(len(test_label))  # the number of error divide the total test label will be error rate
print 'error rate: ', e_rate


## Q1-4 ##
# the function about labelencoder, return a dataframe which has contributed to nominal 
def label(self):
    from sklearn.preprocessing import LabelEncoder
    self = self.apply(LabelEncoder().fit_transform)
    return self  

# the function about extracting data from a dataset, return an array of data
def attrib(self):
    data = [list(self.iloc[i,0:13]) for i in range(len(self))]
    data = np.asarray(data)
    return data

# the function about extracting target from a dataset, return a list of target
def target(self):
    target = list(self.iloc[:,13])
    return target

# the function about calculating error rate, return a percentage of error rate
def err(x,y):
    from sklearn import tree
    tree = tree.DecisionTreeClassifier(random_state=0)
    tree.fit(x,y)
    error = 0
    for i, v in enumerate(tree.predict(D_data)):
        if v != D_target[i]:
            error+=1
    erate = error/float(len(D_target))
    return erate

# -- 1st create a dataframe 'temp' which has at least one Na in each instances
n = [idx for idx,row in adult.iterrows()  if all(row.notnull())]  ## the index of the rows without NA 
temp = adult.drop(adult.index[n]) # this dataframe are the rows that includes NaN (3620 rows)

# -- 2nd create a dataframe 'nona' which select rows randomly from the dataframe without NaN.  
n = [idx for idx,row in adult.iterrows()  if any(row.isnull())]  ## the rows that includes NA 
temp1 = adult.drop(adult.index[n]) # this dataframe are the rows without Na (45222 rows)
nona = temp1.sample(n=3620, random_state=0, axis=0)  

# -- 3rd combine two dataframes as Dp (nona: without Na ; temp: instances with Na)
Dp = pd.concat([nona, temp]) # the smaller dataset D'

# -- 4th Dataframe D1' which need to fill na as sting "missing"
D1 = Dp.fillna('missing')

# -- 5th Dataframe D2' which need to fill na with most popular value from each attributes
values1 = {}
for i in range(1,len(Dp.keys())-1):
    v = {Dp.keys()[i] : Dp.iloc[:,i].value_counts().idxmax()}  # v is a dictionary of the most popular value
    values1.update(v)
D2 = Dp.fillna(value=values1)  # fill NaN with the most popular value

# -- 6th Dataframe D without instances from D1' and D2'  (drop the instances in Dp)
n = [idx for idx, row in Dp.iterrows()]  # n is the index of the instances in Dp(the smaller dataset)
D = adult.drop(adult.index[n])  # drop instances from dataframe 'adult' which has index n

# create the training data and testing data for each dataframe
D_data = attrib(label(D)) 
D_target = target(label(D))
D1_data = attrib(label(D1))
D1_target = target(label(D1))
D2_data = attrib(label(D2))
D2_target = target(label(D2))

print 'Question 1-4 : '
print "D1' error rate : ", err(D1_data, D1_target)
print "D2' error rate : ", err(D2_data, D2_target)


