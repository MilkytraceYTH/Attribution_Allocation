#!/usr/bin/env python
# coding: utf-8

# In this assignment, we are going to analyze the effectiveness of advertising via different channels by calculating CAC and marginal CAC for each channel, and ultimately decide how much of our budget to allocate to each channel.

# In[21]:


# import libraries
import pandas as pd
import numpy as np


# In[5]:


df = pd.read_csv('attribution_allocation_student_data.csv')
df1 = pd.read_csv('channel_spend_student_data.csv')


# In[4]:


df.head()


# #### Attribution
# 
# To calculate CAC, we follow the formula:
# $$\frac{TotalCost}{Total Number of  New Customers}$$
# 
# To get the total number of new customers of each channel, we first need to perform attribution modeling for each channel.
# 
# **The 3 models we will use in this section are:**
# 1. Last interction
# 2. First interaction
# 3. Linear model 

# ##### Last Interaction
# 
# Useful when our ads and campaigns are designed to attract people at the moment of purchase, or our business is primarily transactional with a sales cycle that does not involve a consideration phase.
# 
# Attributes 100% of the conversion value to the last channel with which the customer interacted before buying or converting

# In[52]:


def last_interaction(df):
    """
    Takes in a dataframe,
    count the number of occurances of each channel as the last touch
    Returns a dataframe that counts the number of occurances of each channel
    """
    touches = ['touch_5','touch_4','touch_3','touch_2','touch_1']
    df['last_touch'] = np.nan # create a new column with all null values
    while df['last_touch'].isnull().sum() != 0: # as long as there are null values for this column
        for touch in touches: # for each of the touches 
            # fill the last_touch column with the value in touch, starting from touch 5
            df['last_touch'].fillna(df[touch],inplace=True) 
    
    return df.groupby('last_touch').count().reset_index().rename(columns={'tier' : 'count'})[['last_touch','count']]


# In[203]:


# only focus on those who have converted
subdf = df[df['convert_TF'] == True]
last_count = last_interaction(subdf)
last_count


# ##### First Interaction
# 
#  This model is appropriate if you run ads or campaigns to create initial awareness. For example, if your brand is not well known, you may place a premium on the keywords or channels that first exposed customers to the brand.
#  
#  The First Interaction model attributes 100% of the conversion value to the first channel with which the customer interacted.
# 

# In[68]:


def first_interaction(df):
    """
    Takes in a dataframe,
    count the number of occurances of each channel as the first touch
    Returns a dataframe that counts the number of occurances of each channel
    """
    
    return df.groupby('touch_1').count().reset_index().rename(columns={'tier' : 'count'})[['touch_1','count']]


# In[204]:


# get first interaction values
first_count = first_interaction(subdf)
first_count


# ##### Linear Model
# 
# This model is useful if your campaigns are designed to maintain contact and awareness with the customer throughout the entire sales cycle. In this case, each touchpoint is equally important during the consideration process.
# 
# The Linear model gives equal credit to each channel interaction on the way to conversion.
# 
# To calculate, we will first need to get the number of touches for each customer (call it n), then assign $\frac{1}{n}$ to each of the channels that touched the customer. 

# In[266]:


def linear_model(df):
    
    # no_of_touches (n) = number of channels that reaches the customer 
    for i, row in df.iterrows():
        df.loc[i,'no_of_touches'] = df.loc[i][touches].notnull().sum()
   
    # assign value to attribution - each touch is assigned equally among the number of touches 
    for touch in touches:
        df[touch+'att'] = df[touch].notnull()/df['no_of_touches']
    
    # create a dictionary for each channel
    linear_dict = {'organic_search':0,'direct':0,'display':0,'email':0,'social':0,'paid_search':0,'referral':0}
    
    # update the dictionary
    for i, row in df.iterrows(): # for each customer
        for touch in touches: # for each touch
            channel = df.loc[i][touch] # get the channel name of each touch

            if str(channel) == 'nan': 
                continue
                
            # if channel name is not null:
            linear_dict[channel] += df.loc[i][touch+'att'] # update the dictionary by attribution value 
   
    # return the dictionary as a dataframe         
    return pd.DataFrame.from_dict(linear_dict,orient='index',columns=['count'])
        


# In[175]:


linear_count = linear_model(subdf)
linear_count.reset_index().rename(columns = {'index':'touch'})
linear_count


# In[174]:


linear_count.reset_index().rename(columns = {'index':'touch'})


# ##### Calculate CAC

# In[193]:


# prepare the total cost column
cost = {'social': 300, 'organic_search': 0, 'referral': 300, 'email': 300, 'paid_search': 300, 'display': 300, 'direct': 0}
cost_df = pd.DataFrame.from_dict(cost,orient='index',columns=['total_cost'])


# **Last Interaction**

# In[205]:


last_count = last_count.merge(cost_df,left_on='last_touch',right_index=True,how='inner')


# In[208]:


last_count['CAC'] = last_count['total_cost']/last_count['count']
last_count


# **First Interaction**

# In[210]:


first_count = first_count.merge(cost_df,left_on='touch_1',right_index=True,how='inner')


# In[212]:


first_count['CAC'] = first_count['total_cost']/first_count['count']
first_count


# **Linear Model**

# In[214]:


linear_count = linear_count.merge(cost_df,left_index=True,right_index=True,how='inner')


# In[219]:


linear_count['CAC'] = linear_count['total_cost']/linear_count['count']
linear_count.reindex(first_count['touch_1'].to_list())


# **Observation**:
# 
# The CAC for the 3 methods turn out to be quite similar. 
# 
# **Conclusion**:
# 
# Based on this calculation, if our objective is to minimize CAC, we should prioritize spending in display ads, because it had the lowest CAC.

# #### Allocation
# 
# In this section, we will calculate the marginal CAC for each channel, for each tier.
# 
# To calculate marginal CAC, we follow the formula:
# $$marginal CAC = \frac{MarginalCost}{MarginalConversion}$$
# 
# In this case, because all of the tiers differ by 50 dollars, the marginal cost between each tier is 50 dollars (except for direct and organic search, which are free channels). Marginal conversion is the difference in conversions between two adjacent tiers. 

# In[222]:


# seperate each tier 
tier_1 = subdf[subdf['tier'] == 1]
tier_2 = subdf[subdf['tier'] == 2]
tier_3 = subdf[subdf['tier'] == 3]


# In[245]:


allocation_df = pd.concat([last_interaction(tier_1),last_interaction(tier_2),last_interaction(tier_3)],axis=1)
allocation_df.columns = ['channel','tier1','na','tier2','na2','tier3']


# In[248]:


allocation_df.drop(['na','na2'],axis=1,inplace=True)


# In[251]:


# get marginal conversions
allocation_df['marginal_2'] = allocation_df['tier2'] - allocation_df['tier1']
allocation_df['marginal_3'] = allocation_df['tier3'] - allocation_df['tier2']


# In[252]:


allocation_df


# In[262]:


# get marginal CAC
allocation_df['marginal spend'] = np.array([0,50,50,0,50,50,50])
allocation_df['M-CAC 1'] = allocation_df['marginal spend']/allocation_df['tier1']
allocation_df['M-CAC 2'] = allocation_df['marginal spend']/allocation_df['marginal_2']
allocation_df['M-CAC 3'] = allocation_df['marginal spend']/allocation_df['marginal_3']


# In[263]:


allocation_df


# In[265]:


allocation_df.to_csv('result.csv')


# **Observations**
# 
