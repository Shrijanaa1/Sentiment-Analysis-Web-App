import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score, confusion_matrix

data = pd.read_csv("Tweets.csv")

#First of all let's drop the columns which we don't required

waste_col = ['tweet_id', 'airline_sentiment_confidence',
     'negativereason_confidence', 'airline_sentiment_gold',
     'name', 'negativereason_gold', 'retweet_count',
     'tweet_coord', 'tweet_location', 'user_timezone']

data.drop(waste_col, axis = 1)

def clean_tweet(text):
  # remove URLs
  text = re.sub(r"http\S+", "", text)

  # remove user mentions
  text = re.sub(r"@\S+", "", text)

  # remove hashtags
  text = re.sub(r"#\S+", "", text)

  # remove emojis
  emoji_pattern = re.compile("["
                             u"\U0001F600-\U0001F64F"  # emoticons
                             u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                             u"\U0001F680-\U0001F6FF"  # transport & map symbols
                             u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                             "]+", flags=re.UNICODE)
  text = emoji_pattern.sub(r'', text)

  # remove numbers and punctuation
  text = re.sub(r"[^a-zA-Z]", " ", text)

  # convert to lowercase
  text = text.lower()

  # tokenize words
  words = word_tokenize(text)

  # remove stopwords
  stop_words = set(stopwords.words('english'))
  filtered_words = [word for word in words if word not in stop_words]

  # lemmatize words
  lemmatizer = WordNetLemmatizer()
  lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

  # join words back into a string
  cleaned_tweet = " ".join(lemmatized_words)

  return cleaned_tweet

# apply preprocessing to the tweet text
data['clean_text'] = data['text'].apply(clean_tweet)

# Pre-processing: Convert text to numerical features
tfidf = TfidfVectorizer(min_df=5, max_df=0.95)
X = tfidf.fit_transform(data["clean_text"])
y = data["airline_sentiment"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training and evaluation
# Support Vector Machine (SVM)
svm = SVC(kernel='linear', degree=5)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
cm_svm = confusion_matrix(y_test, y_pred_svm)
print("SVM accuracy:", acc_svm)
print("SVM f1 score:", f1_svm)
print("SVM recall score:", recall_svm)
print("SVM confusion matrix:\n", cm_svm)

# Multinomial Naive Bayes (MNB)
mnb = MultinomialNB()
mnb.fit(X_train.toarray(), y_train)
y_pred_mnb = mnb.predict(X_test.toarray())
acc_mnb = accuracy_score(y_test, y_pred_mnb)
f1_mnb = f1_score(y_test, y_pred_mnb, average='weighted')
recall_mnb = recall_score(y_test, y_pred_mnb, average='weighted')
cm_mnb = confusion_matrix(y_test, y_pred_mnb)
print("MNB accuracy:", acc_mnb)
print("MNB f1 score:", f1_svm)
print("MNB recall score:", recall_svm)
print("MNB confusion matrix:\n", cm_svm)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("RF accuracy:", acc_rf)
print("RF f1 score:", f1_rf)
print("RF recall score:", recall_rf)
print("RF confusion matrix:\n", cm_rf)

# Save the trained model
import joblib
joblib.dump(svm, "svm_model.pkl")
joblib.dump(tfidf, 'Vectorizer.pkl')
