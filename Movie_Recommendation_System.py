#!/usr/bin/env python
# coding: utf-8

# # Movie Recommendation System

# **Name:**  Riddhi Mahesh Dange

# **Download the dataset from here:** https://grouplens.org/datasets/movielens/1m/

# In[1]:


# Import all the required libraries
import numpy as np
import pandas as pd


# ## Reading the Data
# 

# In[4]:


# Read the dataset from the two files into ratings_data and movies_data
column_list_ratings = ["UserID", "MovieID", "Ratings","Timestamp"]
ratings_data  = pd.read_csv('ratings.dat',sep='::',names = column_list_ratings, encoding= "latin-1",engine="python")
column_list_movies = ["MovieID","Title","Genres"]
movies_data = pd.read_csv('movies.dat',sep = '::',names = column_list_movies, encoding="latin-1", engine="python")
column_list_users = ["UserID","Gender","Age","Occupation","Zixp-code"]
user_data = pd.read_csv("users.dat",sep = "::",names = column_list_users, encoding="latin-1", engine="python")


# `ratings_data`, `movies_data`, `user_data` corresponds to the data loaded from `ratings.dat`, `movies.dat`, and `users.dat` in Pandas.

# ## Data analysis

# In[5]:


data=pd.merge(pd.merge(ratings_data,user_data),movies_data)
data


# Next, we can create a pivot table to match the ratings with a given movie title. Using `data.pivot_table`, we can aggregate (using the average/`mean` function) the reviews and find the average rating for each movie. We can save this pivot table into the `mean_ratings` variable. 

# In[6]:


mean_ratings=data.pivot_table('Ratings','Title',aggfunc='mean')
mean_ratings


# Now, we can take the `mean_ratings` and sort it by the value of the rating itself. Using this and the `head` function, we can display the top 15 movies by average rating.

# In[7]:


mean_ratings=data.pivot_table('Ratings',index=["Title"],aggfunc='mean')
top_15_mean_ratings = mean_ratings.sort_values(by = 'Ratings',ascending = False).head(15)
top_15_mean_ratings


# In[8]:


mean_ratings=data.pivot_table('Ratings',index=["Title"],columns=["Gender"],aggfunc='mean')
mean_ratings


# We can now sort the ratings as before, but instead of by `Rating`, but by the `F` and `M` gendered rating columns. Print the top rated movies by male and female reviews, respectively.

# In[9]:


data=pd.merge(pd.merge(ratings_data,user_data),movies_data)

mean_ratings=data.pivot_table('Ratings',index=["Title"],columns=["Gender"],aggfunc='mean')
top_female_ratings = mean_ratings.sort_values(by='F', ascending=False)
print(top_female_ratings.head(15))

top_male_ratings = mean_ratings.sort_values(by='M', ascending=False)
print(top_male_ratings.head(15))


# In[10]:


mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_values(by='diff')
sorted_by_diff[:10]


# Grouping the data-frame, instead, to see how different titles compare in terms of the number of ratings. Group by `Title` and then take the top 10 items by number of reviews. We can see here the most popularly-reviewed titles.

# In[11]:


ratings_by_title=data.groupby('Title').size()
ratings_by_title.sort_values(ascending=False).head(10)


# In[12]:


filtered_data= ratings_by_title.groupby('Title').filter(lambda x:(x>=2500).all())
filtered_data

Similarly, we can filter our grouped data-frame to get all titles with a certain number of reviews. Filter the dataset to get all movie titles such that the number of reviews is >= 2500.
# Creating a ratings matrix using Numpy. This matrix allows us to see the ratings for a given movie and user ID. Every element $[i,j]$ is a rating for movie $i$ by user $j$. Print the **shape** of the matrix produced.
# 

# In[13]:


# Create the matrix
### use numpy to create a ratings data matrix
nr_users = np.max(ratings_data.UserID.values)
nr_movies = np.max(ratings_data.MovieID.values)
ratings_matrix = np.ndarray(shape=(nr_users, nr_movies),dtype=np.uint8)
ratings_matrix[ratings_data.UserID.values - 1, ratings_data.MovieID.values - 1] = ratings_data.Ratings.values


# In[14]:


# Print the shape
ratings_matrix.shape


# In[15]:


ratings_matrix


# Normalizing the ratings matrix using Z-score normalization. While we can't use `sklearn`'s `StandardScaler` for this step, we can do the statistical calculations ourselves to normalize the data.

# In[16]:


print(data.isna().sum())


# In[17]:


# ratings_col_average = np.mean(ratings_matrix, axis = 0)
ratings_col_average = ratings_matrix.mean(axis = 0)
print(ratings_col_average)
ratings_matrix = (ratings_matrix - ratings_col_average)


# In[18]:


ratings_matrix = (ratings_matrix - ratings_matrix.mean(axis = 0))/ratings_matrix.std(axis = 0)
ratings_matrix[np.isnan(ratings_matrix)] = 0


# In[19]:


ratings_matrix.shape


# In[20]:


ratings_matrix


# We're now going to perform Singular Value Decomposition (SVD) on the normalized ratings matrix.

# In[21]:


# Compute the SVD of the normalised matrix
U, S, V = np.linalg.svd(ratings_matrix)


# In[22]:


# Print the shapes
print("Shape of U is", U.shape)
print("Shape of S is", S.shape)
print("Shape of V is", V.shape)


# Reconstructing four rank-k rating matrix $R_k$, where $R_k = U_kS_kV_k^T$ for k = [100, 1000, 2000, 3000].

# In[23]:


r_1000 = None
for k in [100, 1000, 2000, 3000]:
  u_k = np.matrix(U[:, :k])
  s_k = np.diag(S[:k])
  v_k = np.matrix(V[:k, :])
  r_k = np.dot(np.dot(u_k, s_k), v_k)
  print("R", k, "shape:", r_k.shape)
  print(r_k)
  if k == 1000:
    r_1000 = r_k


# ### Cosine Similarity
# Cosine similarity is a metric used to measure how similar two vectors are. Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space. Cosine similarity is high if the angle between two vectors is 0, and the output value ranges within $cosine(x,y) \in [0,1]$. $0$ means there is no similarity (perpendicular), where $1$ (parallel) means that both the items are 100% similar.
# 
# $$ cosine(x,y) = \frac{x^T y}{||x|| ||y||}  $$

# **Based on the reconstruction rank-1000 rating matrix $R_{1000}$ and the cosine similarity,** sorting the movies which are most similar. Using Function `top_cosine_similarity` which sorts data by its similarity to a movie with ID `movie_id` and returns the top $n$ items, and a second function `print_similar_movies` which prints the titles of said similar movies. Return the top 5 movies for the movie with ID `1377` (*Batman Returns*):

# In[24]:


def top_cosine_similarity(data, movieID, topN = 5):
  x_t = data[:, movieID - 1]
  y = data
  magnitude_x = np.linalg.norm(x_t)
  magnitude_y = np.linalg.norm(y)
  cosineSimilarity = np.dot(x_t, y)/ (magnitude_x * magnitude_y)
  topSortedIndices = np.argsort(-cosineSimilarity)
  returnIndices = topSortedIndices[1: topN + 1]
  return returnIndices

def print_similar_movies(movie_data,movieID,top_indexes):
  print('Most Similar movies: ')
  for id in top_indexes + 1:
      print(movie_data[movie_data["MovieID"] == id]["Title"].values[0])


# In[25]:


k = 1000
movie_id = 1377
top_n = 5

ydata = V[:k, :]
indexes = top_cosine_similarity(ydata, movie_id, top_n)
print_similar_movies(movies_data, movie_id, indexes)

