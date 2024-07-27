# importing necessary packages
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import nltk
warnings.filterwarnings("ignore", category=DeprecationWarning)

# %matplotlib inline
# train dataset used for our analysis
train = pd.read_csv('train.csv')
train_original=train.copy()                          # making a copy of dataset
# train_original

#test dataset used for our analysis
test = pd.read_csv('test.csv')
test_original=test.copy()
# test_original


# We combine Train and Test datasets for pre-processing stage
frames=[train,test]
combine=pd.concat(frames,ignore_index=True,sort=True)
# combine = train.append(test,ignore_index=True,sort=True)


#---------------------------------Data Preprocessing-----------------------------------


#Removing Twitter Handles (@user)
# a user-defined function to remove unwanted text patterns from the tweets.
# It takes two arguments, one is the original string of text and the other is the pattern of text that we want to remove from the string.
# The function returns the same input string but without the given pattern.
# We will use this function to remove the pattern ‘@user’ from all the tweets in our data.
def remove_pattern(text, pattern):
    # re.findall() finds the pattern i.e @user and puts it in a list for further task
    r = re.findall(pattern, text)                   #re is regular expression library here
    # re.sub() removes @user from the sentences in the dataset
    for i in r:
        text = re.sub(i, "", text)

    return text


# creating another column Tidy_Tweets in which @<user> is removed
combine['Tidy_Tweets'] = np.vectorize(remove_pattern)(combine['tweet'], "@[\w]*")


# Removing Punctuations, Numbers, and Special Characters
# Punctuations, numbers and special characters do not help much. It is better to remove them from the text just as we removed the twitter handles.
# Here we will replace everything except characters and hashtags with spaces.
combine['Tidy_Tweets'] = combine['Tidy_Tweets'].str.replace("[^a-zA-Z#]", " ")


# Removing Short Words
# We have to be a little careful here in selecting the length of the words which we want to remove.
# So, I have decided to remove all the words having length 3 or less. For example, terms like “hmm”, “oh” are of very little use.
# It is better to get rid of them.
combine['Tidy_Tweets'] = combine['Tidy_Tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

#Tokenization (Breaking sentences into words)
tokenized_tweet = combine['Tidy_Tweets'].apply(lambda x: x.split())


#Stemming
from nltk import PorterStemmer
ps = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])

#Now let’s stitch these tokens back together.
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])



combine['Tidy_Tweets'] = tokenized_tweet

# Tfidf to data frame
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words='english')
tfidf_matrix=tfidf.fit_transform(combine['Tidy_Tweets'])
df_tfidf = pd.DataFrame(tfidf_matrix.todense())


#----------------------------Model Training-----------------------------------------------

train_tfidf_matrix = tfidf_matrix[:31962]
train_tfidf_matrix.todense()

#Splitting the data set into training and validation dataset
from sklearn.model_selection import train_test_split
x_train_tfidf,x_valid_tfidf,y_train_tfidf,y_valid_tfidf = train_test_split(train_tfidf_matrix,train['label'],test_size=0.3,random_state=17)



from sklearn.linear_model import LogisticRegression
sentiment_model = LogisticRegression(random_state=0,solver='lbfgs')
# Fitting the Logistic Regression Model

sentiment_model.fit(x_train_tfidf,y_train_tfidf)
# The first part of the list is predicting probabilities for label:0
# and the second part of the list is predicting probabilities for label:1


import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


stopword=set(stopwords.words('english'))
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [ps.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

def predict(sample_text):
    data=clean(sample_text)
    data=tfidf.transform([sample_text]).toarray()
    return sentiment_model.predict(data)[0]