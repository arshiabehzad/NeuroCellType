#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import seaborn as sns
import numpy as np


# In[2]:


df = pd.read_csv("cell_types_specimen_details.csv")


# In[3]:


df.head()


# In[7]:



new_df = df.filter(['specimen__id','specimen__hemisphere','structure__name','ef__avg_firing_rate','tag__dendrite_type','structure_name', 'donor_species'], axis=1)


# In[13]:


pd.set_option('display.max_columns', 100)
new_df.head()


# In[10]:


new_df.info()


# In[ ]:




