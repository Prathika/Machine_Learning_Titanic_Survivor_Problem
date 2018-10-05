
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train_data = pd.read_csv(‘train.csv')
test_data = pd.read_csv(‘test.csv')


# In[2]:


train_data.head()
# The train data has survived column (which is our target attribute), whereas the test data doesn't contain it
# We should remove the survived column from train data later


# In[3]:


test_data.head()


# In[4]:


# Have a glimpse of both training and test data summary
train_data.describe()


# In[5]:


test_data.describe()


# In[6]:


# Purely going with the gender for prediction first
# With the both dataset we could see that age has missing columns
# We can either drop out them or fill the missing values with mean value
import seaborn as sns
sns.countplot(data=train_data,x='Sex',hue='Survived')


# In[7]:


train_data['Age'].fillna((train_data['Age'].mean()), inplace=True)


# In[8]:


train_data.describe(include='all')


# In[9]:


# Survivor based on Pclass
sns.countplot(data=train_data,x='Pclass',hue='Survived')


# In[10]:


# Fill the missing age values with mean value
test_data['Age'].fillna((test_data['Age'].mean()), inplace=True)


# In[11]:


test_data.describe(include='all')


# In[12]:


train_data['Family'] =  train_data["Parch"] + train_data["SibSp"]


test_data['Family'] =  test_data["Parch"] + test_data["SibSp"]


# drop Parch & SibSp
train_data = train_data.drop(['SibSp','Parch'], axis=1)
test_data = test_data.drop(['SibSp','Parch'], axis=1)
train_data


# In[13]:


# Convert the string (gender) to numeric value
gender = {'male':0, 'female':1}
train_data['Sex'] = train_data['Sex'].apply(lambda x: gender.get(x))
test_data['Sex'] = test_data['Sex'].apply(lambda x: gender.get(x))


# In[14]:


#columns_to_drop = ['Name','SibSp','Parch','Ticket','Fare','Cabin']
# The below columns can be dropped, just trying to predict the survival based on Age,Sex and Pclass for now
# columns_to_drop = ['Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
columns_to_drop = ['Name','Ticket','Fare','Cabin','Embarked']
X_train = train_data.drop(columns_to_drop+['Survived'],axis=1)
Y_train = train_data['Survived']
X_test = test_data.drop(columns_to_drop, axis=1)


# In[15]:


X_test.head()


# In[16]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(max_depth=10,min_samples_split=2,n_estimators=100 , random_state=1 )
rf_model = forest.fit(X_train,Y_train)

my_prediction = rf_model.predict(X_test)

my_solution = pd.DataFrame(my_prediction, X_test.PassengerId, columns = ["Survived"])

my_solution.to_csv("titanic_own_soln_family_size.csv", index_label = ["PassengerId"])


# In[17]:


X_test.shape


# In[18]:


X_train.shape


# In[19]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(max_depth=10,min_samples_split=2,n_estimators=100 , random_state=1 )
rf_model = forest.fit(X_train,Y_train)


# In[20]:


print(rf_model.feature_importances_)


# In[21]:


rf_model.score(X_train, Y_train)


# In[22]:


Y_train.shape

