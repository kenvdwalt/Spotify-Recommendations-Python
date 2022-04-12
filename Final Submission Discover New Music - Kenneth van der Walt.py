#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df2 = pd.read_json('C://Users/K.VanDerWalt/Python for Data Science/spotify_raw/KV/StreamingHistory0.json')
df2.head()


# In[3]:


df3 = pd.read_json('C://Users/K.VanDerWalt/Python for Data Science/spotify_raw/KV/StreamingHistory1.json')
df3.head()


# In[4]:


df = df2.append(df3)
df.head()


# In[5]:


df.describe()


# In[6]:


len(df.index)


# In[7]:


artists = df.groupby('artistName').sum('msPlayed')
artists = artists.sort_values('msPlayed', ascending = False)
artists


# In[8]:


genres1 = pd.read_csv('C://Users/K.VanDerWalt/Python for Data Science/mtv_artists_genres/10000-MTV-Music-Artists-page-1.csv')
genres2 = pd.read_csv('C://Users/K.VanDerWalt/Python for Data Science/mtv_artists_genres/10000-MTV-Music-Artists-page-2.csv')
genres3 = pd.read_csv('C://Users/K.VanDerWalt/Python for Data Science/mtv_artists_genres/10000-MTV-Music-Artists-page-3.csv')
genres4 = pd.read_csv('C://Users/K.VanDerWalt/Python for Data Science/mtv_artists_genres/10000-MTV-Music-Artists-page-3.csv')
genres = genres1.append([genres2, genres3, genres4])
genres.head()


# In[9]:


genres = genres.rename(columns={"name":"artistName"})
genres.head(1)


# In[10]:


genres['artistName'] = genres['artistName'].str.strip()
genres.head(1)


# In[11]:


artists = artists.merge(genres, on = 'artistName', how='inner')
artists.head(1)
# df_excellent_genre_avg_rating.merge(df_satisfactory_genre_avg_rating, on='genres', how='outer')


# In[12]:


top_genres = artists.groupby('genre').sum('msPlayed')
top_genres = top_genres.sort_values('msPlayed', ascending = False)
top_genres.head(10)


# This shows some of the top genres of all time. It's important to now dig deeper and get an idea of trends by genre. This will help inform the technique for a recommender system.

# In[13]:


top_genres['msPlayed'] = top_genres['msPlayed']/3.6e+6


# In[14]:


top_genres.head(10).plot.barh(y='msPlayed',figsize=(15,10))


# In[15]:


df.info()


# In[16]:


df['endTime'] = pd.to_datetime(df['endTime'], infer_datetime_format=True).dt.date
df


# In[17]:


df['week'] = pd.to_datetime(df['endTime']).dt.to_period('W')
df.head()


# In[18]:


hours_per_week = df.groupby('week').sum('msPlayed').sort_values('week', ascending = True)
hours_per_week.head(1)


# In[19]:


hours_per_week['msPlayed'] = hours_per_week['msPlayed']/3.6e+6


# In[20]:


hours_per_week.plot.line()


# In[21]:


df = df.merge(genres, on = 'artistName', how='inner')
df.head(1)


# In[22]:


df = df.dropna(subset=['genre'])
df


# In[23]:


df['msPlayed'] = df['msPlayed']/60000


# In[24]:


df = df[df['genre'].str.contains('Rock|Pop|Alternative|Country|Singer/Songwriter|Electronic|Jazz|Hip-Hop/Rap|Alternative Folk|Folk-Rock', na = False)]
df.head()


# In[25]:


pop = df.loc[df['genre'] == 'Pop']
pop.head(1)


# In[26]:


pop = pop.groupby(['week', 'genre']).sum('msPlayed').sort_values('week', ascending = True)
pop.head(1)


# In[27]:


pop.plot.line(figsize=(10,5))


# In[28]:


df_summary = df.groupby(['week', 'genre']).sum('msPlayed').sort_values('week', ascending = True)
df_summary.head(1)


# In[29]:


df_summary = df_summary.unstack(level=1)
df_summary.head(5)


# In[30]:


df_summary.plot.line(subplots=True, figsize=(15,200))


# Let's create a hybrid recommender system as follows:
# 
# 1) Find the genres with the largest increase for recent activity
# 
# 2) Of those genres, find the top recent artists
# 
# 3) The user can go to spotify's artists radio for those artists and discover new music
# 
# 4) This addresses a niche in Spotify's recommendations, since in the app there's no straightforward way to work out one's recent top artists. It does have a section for top genres, however the resulting content is not specifically tailored to the user, in the way that "discover weekly" is.
# 
# 5) So this combines more macro trends of top genres, which are less immediately obvious to the user (it's quite easy to recall artists one has been interested in recently). It groups together multiple artists and finds overall genres that the user has been gravitating towards. Then, by giving the top artists of that genre, it allows Spotify to continue the heavy lifting of a recommender system via "go to artist radio"

# In[31]:


df_summary = df.groupby(['endTime', 'genre']).sum('msPlayed').sort_values('endTime', ascending = True)
df_summary.head(1)


# In[32]:


import datetime 
old = df_summary.loc[datetime.date(year=2021,month=1,day=1):datetime.date(year=2022,month=1,day=1)]
old.head(1)


# In[33]:


old = old.groupby('genre').mean('msPlayed').sort_values('msPlayed', ascending = False)
old.head(10)


# In[34]:


recent = df_summary.loc[datetime.date(year=2022,month=1,day=1):datetime.date(year=2023,month=1,day=1)]
recent.head(1)


# In[35]:


recent = recent.groupby('genre').mean('msPlayed').sort_values('msPlayed', ascending = False)
recent.head(5)


# In[36]:


changes = old.merge(recent, on='genre', how='outer')
changes.head(5)


# In[37]:


changes['change'] = changes['msPlayed_y'] - changes['msPlayed_x']
changes = changes.sort_values('change', ascending = False)
changes = changes.head(5)
changes


# In[38]:


top_genre_artists = df.merge(changes, on='genre', how='inner')
top_genre_artists.head(1)


# In[39]:


top_genre_artists = top_genre_artists.groupby(['artistName', 'genre']).sum('msPlayed').sort_values('msPlayed', ascending = False)
top_genre_artists.head(10)

