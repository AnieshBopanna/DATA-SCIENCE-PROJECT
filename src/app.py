import streamlit as st
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

data = pd.read_csv('deployment_data.csv')
print(data.columns)
data = data.drop(columns=["Unnamed: 0","is_english","contains_url","is_spam","polarity","subjectivity"],axis=1)


def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F" 
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F700-\U0001F77F" 
                               u"\U0001F780-\U0001F7FF" 
                               u"\U0001F800-\U0001F8FF" 
                               u"\U0001F900-\U0001F9FF"  
                               u"\U0001FA00-\U0001FA6F"  
                               u"\U0001FA70-\U0001FAFF" 
                               u"\U00002702-\U000027B0"  
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
pass

def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra space
    text = re.sub(r'\s+', ' ', text)
    
    # Replace repetitions of punctuation
    text = re.sub(r'([!?,.:;"\(\)\[\]])\1+', r'\1', text)
    
    # Remove emojis
    text = remove_emojis(text)
    
    # Remove contractions
    contractions = {
        "y'all": "you all",
        "I'm": "I am",
        "here's": "here is",
        "you're": "you are",
        "that's": "that is",
        "he's": "he is",
        "it's": "it is",
        "she's": "she is",
        "we're": "we are",
        "they're": "they are",
        "I'll": "I will",
        "we'll": "we will",
        "you'll": "you will",
        "it'll": "it will",
        "he'll": "he will",
        "she'll": "she will",
        "I've": "I have",
        "should've": "should have",
        "you've": "you have",
        "could've": "could have",
        "they've": "they have",
        "I'd": "I would",
        "we've": "we have",
        "they'd": "they would",
        "you'd": "you would",
        "we'd": "we would",
        "he'd": "he would",
        "she'd": "she would",
        "didn't": "did not",
        "don't": "do not",
        "doesn't": "does not",
        "can't": "cannot",
        "isn't": "is not",
        "aren't": "are not",
        "shouldn't": "should not",
        "couldn't": "could not",
        "wouldn't": "would not",
        "hasn't": "has not",
        "wasn't": "was not",
        "won't": "will not",
        "weren't": "were not",
        
    }
   
    words = text.split()
    expanded_words = [contractions[word] if word in contractions else word for word in words]
    text = ' '.join(expanded_words)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Initialize the Lemmatizer 
    lemmatizer = WordNetLemmatizer()  
    
    # Apply stemming or lemmatization
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = []
    for token in tokens:
        cleaned_token = lemmatizer.lemmatize(token)
        cleaned_tokens.append(cleaned_token)
    
    # Join the cleaned tokens back to form the cleaned text
    cleaned_text = ' '.join(cleaned_tokens)
    
    return cleaned_text
pass

# Apply preprocessing functions
data['cleaned_comment'] = data['comment'].apply(lambda x: clean_text(remove_emojis(x)))

# Split the data into features (X) and labels (y)
X_train = data['cleaned_comment']
y_train = data['sentiment']

# Fit the TF-IDF vectorizer on the training data
vectorizer = TfidfVectorizer(ngram_range=(1, 4), max_df=0.93, min_df=5)
X_train_v = vectorizer.fit_transform(X_train)

# Initialize the logistic regression model
lr = LogisticRegression(solver='liblinear', max_iter=500, tol=0.001, penalty='l1')

# Train the logistic regression model on the training data
lr.fit(X_train_v, y_train)

# Apply sentiment analysis and get predictions
data['predictions'] = lr.predict(X_train_v)

def main():
    st.title('SENTIMENT ANALYSIS APP',)
    st.write('The app performs sentiment analysis on user comments for this Video.')
    
    # Generate the video thumbnail URL using the video ID
    video_id = "T3FC7qIAGZk"  
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"
    st.image(thumbnail_url, caption='Video relating to the below comments', use_column_width=True)
    
    # Display the data
    st.header('Youtube Comments')
    
    st.dataframe(data)

    
    # Mapping dictionary for sentiment labels
    sentiment_mapping = {1: 'Positive', 0: 'Negative',2: 'Neutral'}

    # Replace numeric values with corresponding labels
    data['sentiment_label'] = data['predictions'].map(sentiment_mapping)
    # Display sentiment distribution
    st.header('Sentiment Distribution')
    st.bar_chart(data['sentiment_label'].value_counts())

    # Filter comments by sentiment
    choices = {1: 'Positive', 0: 'Negative',2: 'Neutral'}
    def format(option) :
        return choices[option]
    selected_sentiment = st.selectbox('Select Sentiment', options=list(choices.keys()),format_func=format)
    filtered_comments = data[data['predictions'] == selected_sentiment]
    st.subheader('Comments with Selected Sentiment')
    st.write(filtered_comments['comment'])
    
if __name__ == '__main__':
    main()
