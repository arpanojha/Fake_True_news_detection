import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt

# Importing the dataset
fake = pd.read_csv('Fake.csv')
truth = pd.read_csv('True.csv')

# Cleaning the texts
import re
import nltk    #nltk.download('stopwords') in case you dont have the list
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
#add the vertict column and get a sense of the length of data
k=[0]*len(fake)
print(len(fake),len(k))
m=[1]*len(truth)
print(len(truth),len(m))
truth['Verdict']=1
fake['Verdict']=0
#merge two data to make on data frame
frames = [truth,fake]
dataset = pd.concat(frames)
print(len(dataset))
print(dataset.columns.tolist())
#check for NaN values . 
dataset.isna().sum()
#clean dataset to remove punctuation, make everything lowercase, remove stop words create corpus
for i in range(0, len(dataset)):
  review = re.sub('[^a-zA-Z]', ' ', str(dataset.iloc[i,1])+" "+str(dataset.iloc[i,0]))
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)
print(corpus)
#extract max_features 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 9000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#train svm with rbf kernel
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
#make prediction 
y_pred = classifier.predict(X_test)
#check success
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

#99.4% accuracy  89% accuracy with naaive bayes