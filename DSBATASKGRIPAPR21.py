#!/usr/bin/env python
# coding: utf-8

# # Author- Pande Saransh
# ### Function- Data Science and Business Analytics
# ### The Sparks Foundation
# ### Task 1- Predictive Analysis
### Importing Standard Libraries
# In[24]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.plotting.register_matplotlib_converters()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

### Defining Filepath
# In[43]:


studentscore_filepath=("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
studentscore_data=pd.read_csv(studentscore_filepath)
studentscore_data.head()


# In[44]:


studentscore_data.tail()

# Data Description 
# In[13]:


studentscore_data.describe()

# Plotting 
# In[17]:


plt.figure(figsize=(10,8))
plt.title("Number of Study Hours VS Score")
plt.xlabel('Hours')
plt.ylabel('Scores')
sns.regplot(x=['Hours'], y='Scores')

# Regression Plot with trendline
# In[48]:


sns.lmplot(data=data_score, x='Hours', y='Scores')
plt.figure(figsize=(12,10))

# Analysis and Prediction
# In[71]:


x=(data_score['Hours'].values).reshape(-2,1)
y=(data_score['Scores'].values)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)


# In[72]:


linreg=LinearRegression()
linreg.fit(x_train,y_train)


# In[73]:


print('Intercept Value is:',linreg.intercept_)
print('linear Coefficient is:',linreg.coef_)


# In[76]:


pred= linreg.predict(x_test)


# In[77]:


pred


# In[79]:


linreg.score(x_test, y_test)


# In[81]:


hours=np.array([9.25,1])
hours.reshape(-2,1)
study=linreg.predict(hours.reshape(-2,1))
print("The predicted score for 9.25 hours will be:",study[0])

