#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


file_path = r'C:\Users\Shreya G\Desktop\7th sem\social-media (1).csv'
data = pd.read_csv("social-media.csv")


# In[11]:


data.rename(columns={'UsageDuraiton': 'UsageDuration'}, inplace=True)


# In[12]:


print("Null values:\n", data.isnull().sum())


# In[13]:


X = data[['UsageDuration', 'Age']]
y = data['TotalLikes']


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[16]:


y_pred = model.predict(X_test)


# In[17]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[18]:


print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[19]:


plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Total Likes")
plt.ylabel("Predicted Total Likes")
plt.title("Actual vs Predicted Total Likes")
plt.show()


# In[20]:


print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)


# In[ ]:




