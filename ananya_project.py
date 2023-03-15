# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:12:02 2023

@author: anany
"""
#1

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
cancer = load_breast_cancer()
df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
df.head()

# Split the dataset into training and testing sets
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Train and evaluate the logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_y_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_y_pred)

# Train and evaluate the KNN classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_y_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_y_pred)

# Train and evaluate the Naive Bayes classifier
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_y_pred = nb.predict(X_test)
nb_acc = accuracy_score(y_test, nb_y_pred)

# Compare the results
data = {'Logistic Regression': [lr_acc], 'KNN': [knn_acc], 'Naive Bayes': [nb_acc]}
results_df = pd.DataFrame(data, index=['Accuracy'])
print(results_df)

#Printing the comparison
results_df.to_csv('model_comparison.csv', index=False)

df = pd.read_csv('model_comparison.csv')
print(df)



#2
import nltk
nltk.download()
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Fetch the web page using requests library
url = "https://thegoodnewshub.com/science/als-patient-breaks-record-for-number-of-words-communicated-a-minute-with-a-brain-implant/"
rsp = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(rsp.content, "html.parser")

# Extract the article text
ar = ""
for pgh in soup.find_all("p"):
    ar += pgh.text

# Perform sentiment analysis on the entire article using NLTK
x = SentimentIntensityAnalyzer()
sen = x.polarity_scores(ar)

# Print the individual scores for positive, negative, and neutral sentiment
print("Positive sentiment:", sen['pos'])
print("Negative sentiment:", sen["neg"])
print("Neutral sentiment:", sen["neu"])

# Generate a word cloud based on the article text
wordcloud = WordCloud(width=800, height=800, background_color="white").generate(ar)

# Create a bar chart for the sentiment scores
lb = ["Positive", "Negative", "Neutral"]
vl = [sen["pos"], sen["neg"], sen["neu"]]
plt.bar(lb, vl)
plt.title("Sentiment Scores")
plt.xlabel("Sentiment")
plt.ylabel("Score")
plt.show()

# Create a pie chart for the sentiment scores
plt.pie(vl, labels=lb, autopct="%1.1f%%")
plt.title("Sentiment Scores")
plt.show()

# Display the word cloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
 



#3
# Import necessary libraries and load the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('CC GENERAL.csv')


# Drop unnecessary columns
df.drop(['CUST_ID'], axis=1, inplace=True)

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Scale the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)


# Using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(df_scaled)
labels = kmeans.labels_


df['Cluster'] = labels
df.groupby(['Cluster']).mean()

# Visualize the clusters
plt.figure(figsize=(10,8))
sns.scatterplot(x=df['PURCHASES'], y=df['BALANCE'], hue=labels, palette='Set1')
plt.title('Clusters by Purchases and Balance')
plt.xlabel('Purchases')
plt.ylabel('Balance')
plt.show()
