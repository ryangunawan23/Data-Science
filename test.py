#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/mk-gurucharan/Regression/master/Startups_Data.csv')
df.head(20)


# In[3]:


df.describe()


# In[4]:


#Redefine State's data into integer, based on the order of the states inputted
#New York = 0
#California = 1
#FLorida = 2

df[['State']] = df[['State']].apply(lambda col:pd.Categorical(col).codes)
df.head()


# In[5]:


#Split data into 80:20 ratio for training and testing
train_df = df.sample(frac = 0.8, random_state = 1)
test_df = df.drop(train_df.index)
train_df.head()


# In[6]:


test_df.head()


# In[7]:


train_df_target = train_df["Profit"]
train_df.pop("Profit")

test_df_target = test_df["Profit"]
test_df.pop("Profit")


# In[8]:


test_df.describe()


# In[9]:


train_df.describe()


# In[17]:


jupyter nbconvert "test.ipynb" --to script


# In[ ]:





# In[ ]:




