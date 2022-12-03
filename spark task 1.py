#!/usr/bin/env python
# coding: utf-8

# # The Spark Foundation - GRIP - Data Science and Business Analytics - Dec'2022

# # TASK 1: Prediction using supervised ML

# ### Author : Nikita Kadam

# ### Dataset Used : Student score

# ##### It can be downloaded through the given link - http://bit.ly/w-data

# ### Problem Statements:
# 
#     Predict the percentage of a student based on the no of students.What will be predicted score if a studies for 9.25 hrs?day?

# ### Import necessary libraries

# In[25]:


import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Read the csv dataset as a pandas dataframe

# In[4]:


data = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')


# In[5]:


data


# In[6]:


data.head()


# In[7]:


data.shape


# In[8]:


data.info()


# In[9]:


data.describe()


# In[10]:


data.isnull().sum()


# In[14]:


data.duplicated().sum()


# In[15]:


data.skew()


# ### Checking normal test results

# In[17]:


from scipy.stats import normaltest
normaltest(data['Hours'])


# ### Exploratory Data Analysis

# In[18]:


## computing a, b at 25 and 75 percent resp., upper and lower limit
a = data['Hours'].quantile(0.25)
c = data['Hours'].quantile(0.75)
IQR= c-a
upper_limit=c+(1.5*IQR)
lower_limit=c-(1.5*IQR)


# In[19]:


upper_limit


# In[20]:


lower_limit


# In[22]:


data_outlier_lowerlimit = data[data['Hours']<lower_limit]
data_outlier_lowerlimit


# In[23]:


data_outlier_upperlimit = data[data['Hours']>upper_limit]
data_outlier_upperlimit


# ### Graphical Analysis

# In[26]:


sns.distplot(data['Scores'])


# In[27]:


import seaborn as sns
sns.pairplot(data)


# In[28]:


## ploting regression plot
sns.regplot(x="Hours",y="Scores",data=data)


# In[30]:


## plotting heat map to check correlation
sns.set(rc={'figure.figsize':(6,4)})
sns.heatmap(data.corr(),annot=True)


# ### Preparing the Data

# In[31]:


## Independent x and Dependent y features
x= data.iloc[:, :-1].values
y= data.iloc[:, 1].values


# In[33]:


print(y)


# In[34]:


print(x)


# In[44]:


from sklearn.model_selection import train_test_split

## Splitting Data Into Train and test data
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.20, random_state=42)


# In[45]:


print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# ### Linear Regresson Model Training

# In[39]:


## importing linearregression

from sklearn.linear_model import LinearRegression
regression=LinearRegression()


# In[47]:


regression.fit(x_train, y_train)


# In[49]:


print(regression.coef_)


# In[50]:


print(regression.intercept_)


# In[53]:


regression_line= regression.intercept_+regression.coef_*x

## Plotting for the test data
plt.scatter(x,y,c='r')
plt.plot(x, regression_line);
plt.show()


# In[54]:


reg_pred=regression.predict(x_test)
print(reg_pred)


# In[56]:


## Computing Training and Testing Accuracy Scores

print("Training Score:" , regression.score(x_train, y_train)*100)
print("Testing Score:" , regression.score(x_test, y_test)*100)


# In[59]:


## Computing hours
hours=9.25
reg_score_pred= regression.predict([[hours]])
print("Number of study hours = {}".format(hours))
print("Predicted score = {}".format(reg_score_pred[0]))


# ### Performance Metrics

# In[60]:


## computing Mean Squared error, Mean absolute error, Root mean squared error


# In[62]:


## import mean_squared_error, mean_absolute_error

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

## Printing Mean Squared error
print(mean_squared_error(y_test, reg_pred))

## Printing Mean absolute error
print(mean_absolute_error(y_test, reg_pred))

## Printing Mean absolute error
print(np.sqrt(mean_squared_error(y_test, reg_pred)))


# In[63]:


## computing R Square

from sklearn.metrics import r2_score
score=r2_score(y_test,reg_pred)
print (score)


# In[64]:


## Adjusted R Square
1-(1-score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)


# ### Conclusion

# The model performs well to predict score based on the number of study hours.
# The model predicted 92% score if a student studies for 9.25 hours/day
