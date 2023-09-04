## Develpoing a Custom Bulit Sentiment Analysis tool for comment moderation by decoding youtube comments and comparing to a large languge model.This will help content creators on Youtube better understand thier audience's sentiment and also help them moderate the comments and mitigate potential crisis effectively and help maintain their reputation. 

## Overview
This project aims to analyze the sentiment in comments from YouTube videos. And comparing this custom built cost effective model with a large language pre-trained transformer model and provide insights.

## Approach
To accomplish the goals of sentiment analysis and model comparison , we will follow below steps :

1. Data Collection: Utilize the YouTube Data API to extract comments from a selected channel.  Gather relevant information such as comment text, author, timestamp, and video metadata. In my case , i extacted comments from top 20 most viewed videos of the Lex Freidmen podcast. 

2. Data Preprocessing: Applied text cleaning techniques, including removing emojis, punctuation, numbers, and stop words, text to lowercase,expand contractions. 

3. Feature Extraction : Used TF_IDF vectorizer to transform textual data into numerical format suitable for machine learning models.

4. Modelling and Evaluation: Utilize a range of  classifiers to train a sentiment analysis model using the labeled comments. Fine-tune models hyperparameters ,get performance metrics like accuracy,precisionand recall

5. Comparative Analysis : Compared the different metrics like accuracy , recall and precision of my best perfomring model with a large pre trained model and uncovered the pros and cons for each. 

## Technologies Used
- Programming Language: Python
- Libraries/Frameworks: YouTube Data API, Natural Language Processing (NLP) libraries (NLTK, spaCy), Machine Learning libraries (scikit-learn,pandas ) Data Visualization libraries (Matplotlib, Seaborn), STREAMLIT .

## Expected Outcome
- Trained models for sentiment analysis. Got the best performing model for my use case (User comments on Youtube) among a range of classifier models tried
- Compared the results of the best model with a transfomer model 
- Insights into viewer sentiment towards  videos .
- SENTIMENT ANALYSIS APPLICATION
    With just a few clicks, content creators and businesses can now gain instant insights into the sentiments hidden within YouTube comments. This empowers them to make informed decisions about content strategies, engagement initiatives, and community interactions.
 

# Summary 
My main project goal was to create a sentiment analysis tool for moderating comments and also compare the effectiveness of a custom-built best performing  model with a transformer model. Our findings revealed that the custom logistic regression model achieved an impressive 95% accuracy in sentiment prediction, while the transformer model achieved 61%. We delved into the influential words driving these predictions using coefficients. Interestingly, transformers excel in sentiment prediction when contextual understanding is crucial. To showcase our results, we developed an app. This would help content creators in making informed business decisions about their videos 

# CONCULSION 
In conclusion, while transformer models possess advanced language comprehension tailored for context-driven sentiment analysis and exhibit commendable generalization to unexplored data, their drawback of extended processing times and susceptibility to code crashes when analyzing large data cannot be overlooked. Particularly, in the context of popular YouTube channels like the Lex Friedman podcast with an abundance of audience interactions, a simple and faster  approach to comment analysis becomes paramount. We found that custom bulit model is the way to meet the  specific  demands of efficient sentiment analysis. By doing so, we empower content creators to swiftly and effectively respond to offensive content, deftly navigate potential crises, and enhance overall audience engagement.

# Running the application 
- To run the app, you  use the following command in your terminal or command prompt:
 streamlit run app.py



