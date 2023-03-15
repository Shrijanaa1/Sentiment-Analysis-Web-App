import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import joblib
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt

# Load the trained SVM model and TF-IDF vectorizer
svm_model = joblib.load("svm_model.pkl")
tfidf_vectorizer = joblib.load('Vectorizer.pkl')

# Define a function for cleaning the tweet text
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

def generate_wordcloud(text, sentiment):
   cleaned_word = " ".join([word for word in text.split()
                           if 'http' not in word
                               and not word.startswith('@')
                               and word != 'RT'
                           ])
   if sentiment == "positive":
       stopwords = set(STOPWORDS)
   elif sentiment == "negative":
       stopwords = set(STOPWORDS).union(set(['not', 'no', 'but', 'however', 'although']))
   else:
       stopwords = set(STOPWORDS)
   wordcloud = WordCloud(stopwords=stopwords,
                     background_color='black',
                     width=3000,
                     height=2500
                    ).generate(cleaned_word)
   st.image(wordcloud.to_array(), caption=sentiment.capitalize() + " Wordcloud", use_column_width=True)

def generate_bargraph(data):
   fig, ax = plt.subplots()
   data['predicted_sentiment'].value_counts().plot(kind='bar', ax=ax)
   ax.set_xlabel('Sentiment')
   ax.set_ylabel('Count')
   st.pyplot(fig)

# Define the Streamlit app
def app():
   st.set_page_config(page_title="Airline Tweets Sentiment Analysis App")

   # Display the title and a brief description
   st.title("Airline Tweets Sentiment Analysis App")
   st.write("This app uses Support Vector Machine model to classify the sentiment of tweets as positive, negative, or neutral.")

   # Allow the user to select the input method
   input_method = st.radio("Select Input Method:", ("Enter a tweet", "Upload a CSV file"))

   if input_method == "Enter a tweet":
       # Allow the user to enter a tweet
       tweet = st.text_input("Enter a tweet:")
       if tweet:
           # Clean the tweet text
           cleaned_tweet = clean_tweet(tweet)

           # Convert the cleaned tweet text to a numerical feature vector
           X = tfidf_vectorizer.transform([cleaned_tweet])

           # Make a prediction using the trained model
           y_pred = svm_model.predict(X)

           # Display the predicted sentiment
           st.write("Predicted Sentiment:", y_pred[0])

   elif input_method == "Upload a CSV file":
       uploaded_file = st.file_uploader("Upload a CSV file:", type=["csv"])
       if uploaded_file is not None:
           # Read the CSV file into a pandas DataFrame
           data = pd.read_csv(uploaded_file)
           st.write("### Sample of the uploaded data")
           st.dataframe(data.head())

           # Clean the text of each tweet in the DataFrame
           data['text'] = data['text'].apply(clean_tweet)

           # Use the trained SVM model and TF-IDF vectorizer to predict sentiment for each tweet
           X_test = tfidf_vectorizer.transform(data['text'])
           y_pred = svm_model.predict(X_test)

           # Display the predicted sentiment for each tweet in the DataFrame
           data['predicted_sentiment'] = y_pred
           st.write("### Predicted Sentiment for Uploaded Tweets")
           st.dataframe(data[['text', 'predicted_sentiment']])

           # Generate a wordcloud for the uploaded tweets
           positive_tweets = data[data['predicted_sentiment'] == 'positive']['text'].tolist()
           negative_tweets = data[data['predicted_sentiment'] == 'negative']['text'].tolist()

           st.write("### Wordcloud for Positive Tweets")
           generate_wordcloud(" ".join(positive_tweets), "positive")

           st.write("### Wordcloud for Negative Tweets")
           generate_wordcloud(" ".join(negative_tweets), "negative")

           # Generate a bar graph for the predicted sentiment
           st.write("### Bar Graph for Predicted Sentiment")
           generate_bargraph(data)

if __name__ == '__main__':
   app()
