# importing necessary packages
import re
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# %matplotlib inline
print("\n \n \n")
# train dataset used for our analysis
train = pd.read_csv('train.csv')
train_original=train.copy()                          # making a copy of dataset
print("Number of rows and coloumns of train dataset",train.shape)     #returns number of rows and columns of train dataframe
# train_original


#test dataset used for our analysis
test = pd.read_csv('test.csv')
test_original=test.copy()
print("Number of rows and coloumns of test dataset",test.shape,"\n")
# test_original


# We combine Train and Test datasets for pre-processing stage
frames=[train,test]
combine=pd.concat(frames,ignore_index=True,sort=True)
# combine = train.append(test,ignore_index=True,sort=True)
print(combine.head(),"\n")      # will return first 5 rows
print(combine.tail(),"\n")      # will return last 5 rows


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
print("Removed @username from dataset \n",combine.head(),"\n")


# Removing Punctuations, Numbers, and Special Characters
# Punctuations, numbers and special characters do not help much. It is better to remove them from the text just as we removed the twitter handles.
# Here we will replace everything except characters and hashtags with spaces.
combine['Tidy_Tweets'] = combine['Tidy_Tweets'].str.replace("[^a-zA-Z#]", " ")
print(combine.head(10),"\n")


# Removing Short Words
# We have to be a little careful here in selecting the length of the words which we want to remove.
# So, I have decided to remove all the words having length 3 or less. For example, terms like “hmm”, “oh” are of very little use.
# It is better to get rid of them.
combine['Tidy_Tweets'] = combine['Tidy_Tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
print(combine.head(10),"\n")


#Tokenization (Breaking sentences into words)
tokenized_tweet = combine['Tidy_Tweets'].apply(lambda x: x.split())
print("Tokenized Tweets\n",tokenized_tweet.head(),"\n")


#Stemming
from nltk import PorterStemmer
ps = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])
print("Stemmed Tweets \n")
print(tokenized_tweet.head(),"\n")


#Now let’s stitch these tokens back together.
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combine['Tidy_Tweets'] = tokenized_tweet
print(combine.head(),"\n")

#------------------------------------Data Visualization---------------------------------

#Visualization from Tweets
# Wordcloud
# A wordcloud is a visualization wherein the most frequent words appear in large size and the less frequent words appear in smaller sizes.
# Importing Packages necessary for generating a WordCloud
# from wordcloud import WordCloud,ImageColorGenerator
from wordcloud import WordCloud,ImageColorGenerator
from PIL import Image
import requests

# Store all the words from the dataset which are non-racist/sexist
all_words_positive = ' '.join(text for text in combine['Tidy_Tweets'][combine['label']==0])

# We can see most of the words are positive or neutral. With happy, smile, and love being the most frequent ones.
# Hence, most of the frequent words are compatible with the sentiment which is non racist/sexists tweets.

# combining the image with the dataset
Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))

# We use the ImageColorGenerator library from Wordcloud
# Here we take the color of the image and impose it over our wordcloud
image_colors = ImageColorGenerator(Mask)

# Now we use the WordCloud function from the wordcloud library
wc = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(all_words_positive)

# Size of the image generated
plt.figure(figsize=(10,20))

# Here we recolor the words from the dataset to the image's color
# recolor just recolors the default colors to the image's blue color
# interpolation is used to smooth the image generated
plt.imshow(wc.recolor(color_func=image_colors),interpolation="hamming")
plt.axis('off')
plt.show()

# Store all the words from the dataset which are racist/sexist
all_words_negative = ' '.join(text for text in combine['Tidy_Tweets'][combine['label']==1])

# combining the image with the dataset
Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))

# We use the ImageColorGenerator library from Wordcloud
# Here we take the color of the image and impose it over our wordcloud
image_colors = ImageColorGenerator(Mask)

# Now we use the WordCloud function from the wordcloud library
wc = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(all_words_negative)

plt.figure(figsize=(10,20))         # Size of the image generated

# Here we recolor the words from the dataset to the image's color
# recolor just recolors the default colors to the image's blue color
# interpolation is used to smooth the image generated
plt.imshow(wc.recolor(color_func=image_colors),interpolation="gaussian")
plt.axis('off')
plt.show()


# ---------------------------------Feature Extraction---------------------------------------


# Bag of Words to data frame
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# Refer :https://towardsdatascience.com/basics-of-countvectorizer-e26677900f9c for countVectorizer
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combine['Tidy_Tweets'])
df_bow = pd.DataFrame(bow.todense())
print("features from cleaned tweets using bag of words")
print(df_bow)


# Tfidf to data frame
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words='english')
tfidf_matrix=tfidf.fit_transform(combine['Tidy_Tweets'])
df_tfidf = pd.DataFrame(tfidf_matrix.todense())
print("features from cleaned tweets using term frequency and inverse document frequency")
print(df_tfidf)


#----------------------------Model Training-----------------------------------------------
train_bow = bow[:31962]         #Slices the bow matrix from combined to training bow
train_bow.todense()             #converting to dense matrix from sparse matrix

train_tfidf_matrix = tfidf_matrix[:31962]       #Same as bow
train_tfidf_matrix.todense()                    #Same as bow

#Splitting the data set into training and validation dataset
from sklearn.model_selection import train_test_split
x_train_bow,x_valid_bow,y_train_bow,y_valid_bow = train_test_split(train_bow,train['label'],test_size=0.3,random_state=2)
x_train_tfidf,x_valid_tfidf,y_train_tfidf,y_valid_tfidf = train_test_split(train_tfidf_matrix,train['label'],test_size=0.3,random_state=17)


from sklearn.linear_model import LogisticRegression
Log_Reg = LogisticRegression(random_state=0,solver='lbfgs')
# Fitting the Logistic Regression Model

Log_Reg.fit(x_train_bow,y_train_bow)
# The first part of the list is predicting probabilities for label:0
# and the second part of the list is predicting probabilities for label:1
prediction_bow = Log_Reg.predict_proba(x_valid_bow)
print("Hello\n",prediction_bow)
# exit()

from sklearn.metrics import f1_score
# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
prediction_int = prediction_bow[:,1]>=0.3

prediction_int = prediction_int.astype(np.int)
prediction_int

# calculating f1 score
log_bow = f1_score(y_valid_bow, prediction_int)
log_bow

Log_Reg.fit(x_train_tfidf,y_train_tfidf)

prediction_tfidf = Log_Reg.predict_proba(x_valid_tfidf)
prediction_tfidf
prediction_int = prediction_tfidf[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
prediction_int

# calculating f1 score
log_tfidf = f1_score(y_valid_tfidf, prediction_int)
log_tfidf


from xgboost import XGBClassifier
model_bow = XGBClassifier(random_state=22,learning_rate=0.9)
model_bow.fit(x_train_bow, y_train_bow)
# The first part of the list is predicting probabilities for label:0
# and the second part of the list is predicting probabilities for label:1
xgb=model_bow.predict_proba(x_valid_bow)
xgb

# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
xgb=xgb[:,1]>=0.3
# converting the results to integer type
xgb_int=xgb.astype(np.int)
# calculating f1 score
xgb_bow=f1_score(y_valid_bow,xgb_int)
xgb_bow
model_tfidf=XGBClassifier(random_state=29,learning_rate=0.7)
model_tfidf.fit(x_train_tfidf, y_train_tfidf)
# The first part of the list is predicting probabilities for label:0
# and the second part of the list is predicting probabilities for label:1
xgb_tfidf=model_tfidf.predict_proba(x_valid_tfidf)
xgb_tfidf

# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
xgb_tfidf=xgb_tfidf[:,1]>=0.3
# converting the results to integer type
xgb_int_tfidf=xgb_tfidf.astype(np.int)

# calculating f1 score
score=f1_score(y_valid_tfidf,xgb_int_tfidf)
score

from sklearn.tree import DecisionTreeClassifier
dct = DecisionTreeClassifier(criterion='entropy', random_state=1)
dct.fit(x_train_bow,y_train_bow)
dct_bow = dct.predict_proba(x_valid_bow)
dct_bow

# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
dct_bow=dct_bow[:,1]>=0.3
# converting the results to integer type
dct_int_bow=dct_bow.astype(np.int)
# calculating f1 score
dct_score_bow=f1_score(y_valid_bow,dct_int_bow)
dct_score_bow
dct.fit(x_train_tfidf,y_train_tfidf)
dct_tfidf = dct.predict_proba(x_valid_tfidf)
dct_tfidf

# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
dct_tfidf=dct_tfidf[:,1]>=0.3

# converting the results to integer type
dct_int_tfidf=dct_tfidf.astype(np.int)

# calculating f1 score
dct_score_tfidf=f1_score(y_valid_tfidf,dct_int_tfidf)

dct_score_tfidf



Algo=['LogisticRegression(Bag-of-Words)','XGBoost(Bag-of-Words)','DecisionTree(Bag-of-Words)','LogisticRegression(TF-IDF)','XGBoost(TF-IDF)','DecisionTree(TF-IDF)']

score = [log_bow,xgb_bow,dct_score_bow,log_tfidf,score,dct_score_tfidf]
compare=pd.DataFrame({'Model':Algo,'F1_Score':score},index=[i for i in range(1,7)])
compare.T

plt.figure(figsize=(18,5))

sns.pointplot(x='Model',y='F1_Score',data=compare)

plt.title('Model Vs Score')
plt.xlabel('MODEL')
plt.ylabel('SCORE')
plt.show()

test_tfidf = tfidf_matrix[31962:]
test_pred = Log_Reg.predict_proba(test_tfidf)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv('result.csv', index=False)
res = pd.read_csv('result.csv')
res

sns.countplot(train_original['label'])
sns.despine()

# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))
# print(stopword)

sentiment_model = LogisticRegression(solver='liblinear', random_state=0)
sentiment_model.fit(x_train_bow,y_train_bow)

import string

sample = "hate trump black"
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

data=clean(sample)
data=bow_vectorizer.transform([sample]).toarray()
print(sentiment_model.predict(data))


with open('sentiment_model_1', 'wb') as files:
    pickle.dump(sentiment_model, files)
