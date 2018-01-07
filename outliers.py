#Author: SISS
# coding: utf-8

# In[1]:


ls

# In[2]:

pwd

# In[82]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys as sys
from sklearn import preprocessing


# In[93]:


train = pd.read_csv('nassCDS.csv', sep=';')


# In[110]:


df = pd.DataFrame(train, columns= ['id', 'dead', 'seatbelt', 'frontal', 'sex', 'ageOFocc', 'yearacc', 'yearVeh', 'occRole', 'airb_deploy', 'injSeverity'])


# In[94]:


train.describe()


# In[95]:


columns = [train.columns]


# In[96]:


print([columns])


# In[97]:


train[10:20]


# In[98]:


train.count()


# In[104]:


train.info()


# In[105]:


def outliers(ys):
    threshold = 3

    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.where(np.abs(z_scores) > threshold)


# In[113]:


df[10:20]


# In[116]:


yearacc = train.yearacc


# In[133]:


outliers(yearacc)


# In[167]:


##Train

threshold=3
ys=train.yearacc
mean_y = np.mean(yearacc)
print(mean_y)
stdev_y = np.std(yearacc)
print(stdev_y)
z_scores = [(y - mean_y) / stdev_y for y in ys]

print np.mean(z_scores)
print np.max(z_scores)
print np.min(z_scores)

z_scoresmax = abs(np.max(z_scores))
z_scoresmin = abs(np.min(z_scores))


if z_scoresmax > threshold and zscoresmin > threshold:
    print "There's outliers in here"
else:
    print "This variable doesn't have outliers"


# In[ ]:


def outliers(ys): 
    threshold = 3
    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    if z_scores > abs.threshold:
        print y
    else:
        print "This is not an outlier"

