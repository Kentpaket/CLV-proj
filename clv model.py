#!/usr/bin/env python
# coding: utf-8

# # Columns description
# 
# - **InvoiceNo**: A unique identifier for the invoice. An invoice number shared across rows means that those transactions were performed in a single invoice (multiple purchases).
# - **StockCode**: Identifier for items contained in an invoice.
# - **Description**: Textual description of each of the stock items.
# - **Quantity**: The quantity of the item purchased.
# - **InvoiceDate**: Date of purchase.
# - **UnitPrice**: Value of each item.
# - **CustomerID**: Identifier for customer making the purchase.
# - **Country**: Country of customer.

# In[36]:


# Libraries
import numpy as np, pandas as pd, re, scipy as sp, scipy.stats
import plotly.express as px
import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import cluster
import math
sns.set_theme()
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
color = sns.color_palette()


# In[3]:


df_baza = pd.read_excel(io='Online Retail.xlsx')


# #### Basic Data Analysis

# In[4]:


df_baza.info()


# In[5]:


#Formatting Date/Time
df_baza['InvoiceDate'] = pd.to_datetime(df_baza['InvoiceDate'], format = '%m/%d/%Y %H:%M')
df_baza.head()


# In[6]:


label = []
values = []
print(df_baza.columns)
for col in df_baza.columns:
    label.append(col)
    values.append(df_baza[col].isnull().sum())
    print(col, values[-1])
    
print(values)


# ### Data Preperation

# In[27]:


# remove null data for description and customer ID
df_baza.dropna(how='any',inplace=True)
#only interested in UK data
df_baza=df_baza[df_baza['Country']=='United Kingdom']
#remove cancelled orders
df_baza=df_baza[df_baza['InvoiceNo']!='C%']
# Drop incomplete month
df_baza = df_baza.loc[df_baza['InvoiceDate'] < '2011-12-01']
#remove data with negative quantity and price
df_baza=df_baza[(df_baza['Quantity']>0) & (df_baza['UnitPrice']>0)]
#drop duplicates
df_baza.drop_duplicates(inplace = True)
print("Number of duplicated transactions:", len(df_baza[df_baza.duplicated()]))
#create revenue column
df_baza['Revenue']=df_baza['Quantity']*df_baza['UnitPrice']
df_baza.isnull().sum()


# In[28]:


#categorical data
df_baza['CustomerID'] = pd.Categorical(df_baza['CustomerID'].astype(int))
df_baza['StockCode'] = pd.Categorical(df_baza['StockCode'])
#new columns
df_baza['Revenue']=df_baza['Quantity']*df_baza['UnitPrice']
df_baza['month'] = df_baza['InvoiceDate'].dt.month
df_baza['year'] = df_baza['InvoiceDate'].dt.year
df_baza['DayofWeek'] = df_baza['InvoiceDate'].dt.day_name()
df_baza['month_year'] = pd.to_datetime(df_baza[['year', 'month']].assign(Day=1))
df_baza['hour'] = df_baza['InvoiceDate'].dt.hour
df_baza.head(10)


# In[29]:


df_baza.describe().round(2)


# In[30]:


# checking how many unique customer IDs are there

x = df_baza['CustomerID'].nunique()

# printing the value
print("There are {} number of different customers".format(x))


# In[31]:


df_baza.groupby('StockCode').agg({'Quantity': "sum"}).sort_values(by="Quantity", ascending=False).head(5)


# ### Visualisation 

# In[118]:


plot = pd.DataFrame(df_baza.groupby(['month_year'])['InvoiceNo'].count()).reset_index()
plot2 = plot2.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Sunday']).reset_index()
plot2 = pd.DataFrame(df_baza.groupby(['DayofWeek'])['InvoiceNo'].count())
plot3 = pd.DataFrame(df_baza.groupby(['hour'])['InvoiceNo'].count()).reset_index()
plot4 = pd.DataFrame(df_baza.groupby(['month_year'])['Revenue'].mean()).reset_index()
plot5 = pd.DataFrame(df_baza.groupby(['month_year'])['Revenue'].sum()).reset_index()


# In[87]:


ax = sns.lineplot(x="month_year", y="InvoiceNo", data = plot)


# In[110]:


ax = sns.barplot(x="hour", y="InvoiceNo", data = plot3)


# In[111]:


ax = sns.lineplot(x = 'month_year', y='Revenue', data = plot5)


# In[12]:


#Price distribution
plt.subplots(figsize=(10,8))
sns.distplot(df_baza.Quantity[df_baza.Quantity < 50], label='Unit Price').legend()

plt.xlabel('Unit Price')
plt.ylabel('Normalized Distribution')
plt.title('Unit Price Distribution')
plt.show()


# In[13]:


ax = df_baza.groupby('InvoiceNo')['DayofWeek'].unique().value_counts().sort_index().plot(kind ='bar',color=color[0],figsize=(15,6))
ax.set_xlabel('DayofWeek',fontsize=15)
ax.set_ylabel('Number of Orders',fontsize=15)
ax.set_title('Number of orders for different Days',fontsize=15)
ax.set_xticklabels(('Mon','Tue','Wed','Thur','Fri','Sun'), rotation='horizontal', fontsize=15)


# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_height())

# set individual bar lables using above list
total = sum(totals)

# set individual bar lables using above list
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+0.1, i.get_height()+40,             str(round((i.get_height()/total)*100, 2))+'%', fontsize=15,
                fontweight='500', color='#0e0e43')
plt.show()


# ### Clustering process

# In[33]:


# assign each unique customer with certain  information
df_cus = df_baza.groupby('CustomerID').agg({
    'Revenue': sum,
    'InvoiceNo': lambda x: x.nunique(),
})

df_cus.columns = ['TotalRevenue', 'OrderCount']
df_cus['AvgOrderValue'] = df_cus['TotalRevenue']/df_cus['OrderCount']

df_cus


# In[34]:


# normalize data

# ranked values in each columnes 
ranked_df = df_cus.rank(method='first')

# normalize data
norm_df = (ranked_df - ranked_df.mean()) / ranked_df.std()

norm_df


# ### Select the optimal number of clusters with the Elbow Method

# In[37]:


k_range = range(2, 10)

distortions = []
X = norm_df[['TotalRevenue','OrderCount','AvgOrderValue']].values

for n in k_range:
    model = cluster.KMeans(n_clusters=n, random_state=4)
    model.fit_predict(X)
    cluster_assignments = model.labels_
    centers = model.cluster_centers_
    distortions.append(np.sum((X - centers[cluster_assignments]) ** 2))

plt.plot(k_range, distortions)
plt.xlabel("K")
plt.ylabel("Distortion")
plt.show()


#  k=4 is our 'elbow'

# In[38]:


from sklearn.metrics import silhouette_score

silhouettes = []

for n in k_range:
    model = cluster.KMeans(n_clusters=n, random_state=4)
    model.fit(X)
    silhouettes.append(silhouette_score(X, model.labels_))

plt.plot(k_range, silhouettes)
plt.xlabel("K")
plt.ylabel("Silhouette")
plt.show()


# Again, 4 is our optimal point

# In[40]:


# fitting kmeans model
kmeans = KMeans(n_clusters=4).fit(X)
# intergate each customers and its cluster into a dataframe
fourmeans_cluster_df = norm_df.copy()
fourmeans_cluster_df['Cluster'] = kmeans.labels_
fourmeans_cluster_df


# In[44]:


# assign clusters to before-normalized data
df_cus['Cluster'] = kmeans.labels_
print(df_cus.groupby('Cluster').max())


# In[47]:


# Contribution of each cluster
df_cus.groupby('Cluster').agg({
    'TotalRevenue': sum
}).sort_values('TotalRevenue').plot.pie(y='TotalRevenue', autopct='%1.1f%%', startangle=90)

