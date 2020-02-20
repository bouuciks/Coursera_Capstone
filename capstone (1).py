#!/usr/bin/env python
# coding: utf-8

# In[2]:


#this is capstone for data science kernel


# In[15]:


import pandas as pd


# In[16]:


import numpy as np


# In[17]:


print('Hello Capstone Project Course')


# In[18]:


link = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"

pd=pd.read_table(link)
pd


# In[24]:


from bs4 import BeautifulSoup
import requests
website_url = requests.get('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M').text
soup = BeautifulSoup(website_url,'lxml')
print(soup.prettify())
My_table = soup.find('table',{'class':'wikitable sortable'})
a=My_table.find_All('a')
#creating 3 for loops for each column
Postcodes=[]
for codes in Postcodes:
    Postcodes.append(get.a('Postcode'))
Borough=[]
for boroughs in Borough:
    Borough.append(get.a('Borough'))
Neighbourhood=[]
for hoods in Neighbourhood:
    Neighbourhood.append(get.a('Neighbourhood'))
import pandas as pd
df=pd.DataFrame()
df['Postcodes']=Postcodes
df['Borough']=Borough
df['Neighbourhood']=Neighbourhood
print(df)
    


# In[34]:


import pandas as pd
url='https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'

df=pd.read_html(url, header=0)[0]

df.head(20)
df.columns
df = df[df.Borough != 'Not assigned']
df


# In[35]:


new_df = df.groupby(['Postcode', 'Borough']).agg({'Neighbourhood':lambda x:', '.join(x)}).reset_index()


# In[36]:


new_df


# In[39]:


new_df.shape


# In[ ]:




