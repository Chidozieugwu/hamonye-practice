#!/usr/bin/env python
# coding: utf-8

# In[3]:


def Hello():
    print('Hello world!')


# In[6]:


def hello(name):
    print('Hello,', name)
hello('Bob')
hello('sam')


# In[8]:


def func(name, job):
    print(name, 'is a', job)
func('Bob', 'developer')


# In[4]:


def func(name, job):
    print(name, 'is a', job)
func(job='developer', name='Bob')
func(name='Bob', job='developer')


# In[10]:


def func(name, job = 'developer'):
    print(name, 'is a', job)
func('Bob', 'manager')
func('Bob')


# In[11]:


def print_arguments(*args):
    print(args)
print_arguments(1,54,60,8,98,12)


# In[12]:


def print_arguments(**kwargs):
    print(kwargs)
print_arguments(name='Bob', age=25, job='dev')


# In[14]:


def sum(a,b):
    return a + b
x = sum(3, 4)
print(x)


# In[16]:


def func(a,b):
    return a+b, a-b
x = func(3,2)
print(x)


# In[18]:


class Car:
    pass


# In[13]:


class Car:
    wheel = 4
    def __init__(self, colour, style):
        self.colour = colour
        self.style = style
c = Car('Black', 'Sedan')
print(c.style)
print(c.colour)
c.style = 'SUV'
print(c.style)


# In[5]:


days = pd.Series(['Monday', 'Tuesday', 'Wednesday'])
print(days)


# In[6]:


import pandas as pd
import numpy as np


# In[14]:


days_list = np.array(['Monday', 'Tuesday', 'Wednesday'])
numpy_days = pd.Series(days_list)
print(numpy_days)


# In[21]:


days = pd.Series(['Monday', 'Tuesday', 'Wednesday'], index = ['a', 'b', 'c'])
print(days)


# In[22]:


days[0]


# In[23]:


days[1:]


# In[25]:


days['c']


# In[26]:


print(pd.DataFrame())


# In[53]:


df_dict = {'country':['Ghana', 'Kenya', 'Nigeria', 'Togo'], 'capital': ['Accra', 'Nairobi',  'Abuja', 'Lome'], 'Population': [10000, 8500, 35000, 12000], 'Age': [60, 70, 80, 75]}
df = pd.DataFrame(df_dict, index = [2, 4, 6, 8])
print(df)


# In[37]:


df_list = [['Ghana', 'Accra', 10000, 60], ['Kenya', 'Nairobi', 8500, 70], ['Nigeria', 'Abuja', 35000, 80], ['Togo', 'Lome', 12000, 75]]
df1 = pd.DataFrame(df_list, columns = ['Country', 'Capital', 'Population', 'Age'], index = [2, 4, 6, 8])
df


# In[39]:


df.iloc[3]


# In[40]:


df.loc[6]


# In[43]:


df.at[6, 'country']


# In[44]:


df.iat[3,0]


# In[49]:


df['Population'].sum()


# In[51]:


df.mean()


# In[54]:


df.describe()


# In[55]:


df_dict2 = {'Name': ['James', 'Yemen', 'Caro', np.nan],'Profession': ['Researcher', 'Artist', 'Doctor', 'Writer'], 'Experience': [12, np.nan, 10, 8], 'Height': [np.nan, 175, 180, 150]}
new_df = pd.DataFrame(df_dict2)
new_df


# In[56]:


new_df.isnull()


# In[58]:


new_df.dropna()


# In[60]:


new_df.notnull()


# In[61]:


new_df.replace()


# In[4]:


csv_df = pd.read_csv('sample_file.csv')


# In[9]:


url=https://raw.githubusercontent.com/WalePhenomenon/climate_change/master/fuel_ferc1.csv
    print(url)
fuel_data = pd.read_csv(url, error_bad_lines=False)
fuel_data

fuel_data.describe(include='all')


# In[3]:


import pandas as pd


# In[ ]:


csv_df = pd.read_csv('https://github.com/WalePhenomenon/climate_change/blob/master/fuel_ferc1.csv?raw=true')
csv_df.to_csv('https://github.com/WalePhenomenon/climate_change/blob/master/fuel_ferc1.csv?raw=true', index=False)


# In[12]:


url='https://github.com/WalePhenomenon/climate_change/blob/master/fuel_ferc1.csv?raw=true'
fuel_data = pd.read_csv(url, error_bad_lines=False)
fuel_data.describe(include='all')


# In[13]:


fuel_data.isnull().sum()


# In[19]:


fuel_data.groupby('fuel_unit')['fuel_unit'].count()


# In[18]:


fuel_data.groupby('fuel_unit')['fuel_unit'].count()

fuel_data[['fuel_unit']] = fuel_data[['fuel_unit']].fillna(value='mcf')


# In[16]:


fuel_data.isnull().sum()


# In[17]:


fuel_data.groupby('report_year')['report_year'].count()


# In[21]:


fuel_data.groupby('fuel_type_code_pudl').first()


# In[23]:


import seaborn as sns


# In[14]:


import numpy as np


# In[4]:


import matplotlib.pyplot as plt


# In[40]:


plt.figure(figsize=(8,4))
plt.xticks(rotation=45)
fuel_unit = pd.DataFrame({'unit':['BBL', 'GAL', 'GRAMSU', 'KGU', 'MCF', 'MMTU', 'MWDTH', 'MWHTH', 'TON'], 'count':[7998, 84, 464, 110, 11354, 180, 95, 100, 8958]})
sns.barplot(data=fuel_unit, x='unit', y='count')
plt.xlabel('Fuel unit')


# In[39]:


g = sns.barplot(data=fuel_unit, x='unit', y='count')
plt.xticks(rotation=45)
g.set_yscale("log")
g.set_ylim(1, 12000)
plt.xlabel('fuel unit')


# In[19]:


url='https://github.com/WalePhenomenon/climate_change/blob/master/fuel_ferc1.csv?raw=true'
fuel_data = pd.read_csv(url, error_bad_lines=False)
fuel_data.describe(include='all')


# In[41]:


sample_df = fuel_data.sample(n=50, random_state=4)
sns.regplot(x=sample_df["utility_id_ferc1"], y=sample_df["fuel_cost_per_mmbtu"],
           fit_reg=False)


# In[42]:


sns.boxplot(x="fuel_type_code_pudl", y="utility_id_ferc1",
           palette=["m", "g"], data=fuel_data)


# In[48]:


sns.kdeplot(sample_df['fuel_cost_per_unit_burned'], shade=True, color='b')

