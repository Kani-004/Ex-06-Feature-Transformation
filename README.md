# Ex-06-Feature-Transformation

# AIM

To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM

## STEP 1

Read the given Data

## STEP 2

Clean the Data Set using Data Cleaning Process

## STEP 3

Apply Feature Transformation techniques to all the features of the data set

## STEP 4

Save the data to the file

~~~
# CODE


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

df = pd.read_csv("/content/Data_to_Transform.csv")
df

df.head()

df.isnull().sum()

df.info()

df.describe()

df1 = df.copy()

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

~~~

# OUPUT

# Dataset:

![image](https://user-images.githubusercontent.com/129577149/233902734-714c44b8-c06d-4903-a3b5-c758cd8d2d31.png)


# Head:

![image](https://user-images.githubusercontent.com/129577149/233902824-d76a16da-dea0-4622-9b0c-8809af204507.png)


# Null data:

![image](https://user-images.githubusercontent.com/129577149/233902873-78ce9416-32f0-4b96-bb07-856af2e68285.png)

# Information:

![image](https://user-images.githubusercontent.com/129577149/233902906-2e42b99a-f39e-40dc-b181-19d345cccc1c.png)

# Description:

![image](https://user-images.githubusercontent.com/129577149/233902956-ce24cece-d23f-4b75-af9f-8f45a4bfc050.png)

#  Highly Positive Skew:

![image](https://user-images.githubusercontent.com/129577149/233903082-6341afba-1fed-48a8-830c-b49fb50795e7.png)

# Highly Negative Skew:

![image](https://user-images.githubusercontent.com/129577149/233903131-e8af9201-037a-40f5-bf55-921e05dc474e.png)

# Moderate Positive Skew:

![image](https://user-images.githubusercontent.com/129577149/233903205-abd2bdd5-62bb-45eb-a945-2abc245323d7.png)

# Moderate Negative Skew:

![image](https://user-images.githubusercontent.com/129577149/233903323-15b0baaf-8e4c-48a1-a2c4-06898326b25e.png)

# Log of Highly Positive Skew:

![image](https://user-images.githubusercontent.com/129577149/233903386-0a4aa70e-112a-4894-9d99-204e4754f9d6.png)


# Log of Moderate Positive Skew:

![image](https://user-images.githubusercontent.com/129577149/233903431-8df53abf-c87f-40c4-a86a-b2e6a7e2c150.png)

# Reciprocal of Highly Positive Skew:

![image](https://user-images.githubusercontent.com/129577149/233903483-6222b9d5-36cf-4ff5-83f1-4e3dcc777f95.png)

# Square root tranformation:

![image](https://user-images.githubusercontent.com/129577149/233903538-fc7a5874-5abc-4f40-984d-3c42234b7826.png)

# Power transformation of Moderate Positive Skew:

![image](https://user-images.githubusercontent.com/129577149/233903657-3df5b586-a017-4ca9-a358-e71a56b639e5.png)

# Power transformation of Moderate Negative Skew:

![image](https://user-images.githubusercontent.com/129577149/233903898-8ca95c23-e65a-4775-93eb-a761b5472283.png)

# Quantile transformation:

![image](https://user-images.githubusercontent.com/129577149/233903811-1df1f039-617e-41ed-8d7c-b8efc4298f84.png)

# Result
~~~

Thus, Feature transformation is performed and executed successfully for the given dataset

~~~



