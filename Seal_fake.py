import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('all_comps.csv')
# df.head()

#Get the labels
labels = df.label
labels.head()

#Split the datasets
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size = 0.2)

#Initialize a tfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df= 0.7)

#Fit and transform train set, transfom test set

from sklearn.feature_extraction.text import HashingVectorizer
corpus = ['This is the first document.',
          'This document is the second document.',
          'And this is the third one.',
          'Is this the first document?',] 
vectorizer = HashingVectorizer(n_features=2**4) 
decoded= vectorizer.decode(corpus) 
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)

#Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter = 50)
pac.fit(tfidf_train, y_train)

#Predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
pac.fit(f'Accuracy : {round(score*100, 2)}%')

#Build Confusion matrix
confusion_matrix(y_test, y_pred, labels = ['FAKE', 'REAL'])
