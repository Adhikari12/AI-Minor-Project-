import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Load Dataset
df = pd.read_csv("C:\\Users\\SAI GAYATRI\\OneDrive\\Desktop\\project\\spotify dataset.csv")  
# Step 1: Data Preprocessing
print("Initial Data Overview:\n", df.head())

# Checking for missing values
print("Missing Values:\n", df.isnull().sum())

# Handle missing values (e.g., drop or impute)
df.dropna(inplace=True)

# Encode categorical variables (e.g., genres, playlist_name)
label_encoder = LabelEncoder()
df['genre_encoded'] = label_encoder.fit_transform(df['playlist_genre'])
df['playlist_encoded'] = label_encoder.fit_transform(df['playlist_name'])

# Defining numerical features
scaler = StandardScaler()
numerical_features = ['danceability', 'energy', 'tempo', 'valence', 'acousticness']  # Add relevant columns
df_scaled = scaler.fit_transform(df[numerical_features])
df_scaled = pd.DataFrame(df_scaled, columns=numerical_features)

# Combining scaled features with encoded data
df_preprocessed = pd.concat([df_scaled, df[['genre_encoded', 'playlist_encoded']]], axis=1)

# Step 2: Data Analysis and Visualizations
# Visualizing distributions of features
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature], kde=True, bins=30)
    plt.title(f"Distribution of {feature}")
    plt.show()

# Correlation of matrix and heatmap
correlation_matrix = df_preprocessed.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Step 3: Clustering
# Performing K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Choose an appropriate number of clusters
clusters = kmeans.fit_predict(df_scaled)
df['Cluster'] = clusters

# Visualizing clusters using PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_features = pca.fit_transform(df_scaled)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=df['Cluster'], palette="viridis")
plt.title("Clusters of Songs")
plt.show()

# Step 4: Recommendation System (Content-Based Filtering)
# Calculating cosine similarity
song_features = df_scaled.values
similarity_matrix = cosine_similarity(song_features)

# specifying function to recommend songs based on a given song ID
def recommend_songs(song_id, top_n=5):
    similarity_scores = similarity_matrix[song_id]
    similar_song_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]
    return df.iloc[similar_song_indices]

# Example Recommendation
song_id = 10  # Example song index
recommended_songs = recommend_songs(song_id)
print("Recommended Songs:\n", recommended_songs[['song_name', 'artist', 'genre']])

