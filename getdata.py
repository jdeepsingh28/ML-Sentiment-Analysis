import tweepy
import pandas as pd
import re
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import emoji
import datetime
import os
from newsapi import NewsApiClient

def create_data_directory(tag):
    # get current date and time
    current_datetime = datetime.datetime.now()

    # convert datetime to string format (yyyy-mm-dd-H:M AM/PM)
    current_datetime_str = current_datetime.strftime("%Y-%m-%d")

    # create directory name
    dir_name = f"{tag}-{current_datetime_str}"

    # create the directory if it doesn't exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    return dir_name  # return the directory name

contraction_mapping = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "this's": "this is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "here's": "here is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

def expand_contractions(text, contraction_mapping=contraction_mapping):

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)
    
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def remove_tags(text):

    TAG_RE = re.compile(r'<[^>]+>')
    '''Removes HTML tags: replaces anything between opening and closing <> with empty space'''

    return TAG_RE.sub('', text)

def remove_url(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def preprocess_text(sen):
    # Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Get simple pos tag for lemmatization
    def get_simple_pos_tag(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    sen = expand_contractions(sen.lower())

    # Remove html tags
    sentence = remove_tags(sen)
    
     # Remove urls
    sentence = remove_url(sentence)

    # Replace emoticons and emojis with their meaning
    # Add your own emoticons or emojis here
    emoticons = {
        ":)": " happy ",
        ":(": " sad ",
        ":-)": "smile",
        ":-(": "frown",
        ";)": "wink",
        ":D": "laugh",
        ":/": "skeptical",
    }
    for emoticon, replacement in emoticons.items():
        sentence = sentence.replace(emoticon, replacement)
    sentence = emoji.demojize(sentence, delimiters=(" ", " "))

    # Replace negations (e.g., "not") with "not_"
    negations = ["not", "n't", "no"]
    for negation in negations:
        sentence = sentence.replace(negation, "not_")

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal and lemmatization
    sentence = [lemmatizer.lemmatize(word, get_simple_pos_tag(pos))
                for word, pos in pos_tag(word_tokenize(sentence))
                if len(word) > 1]

    # Join words back into sentence
    sentence = " ".join(sentence)

    # Remove stopwords
    stopwords_list = set(stopwords.words('english'))
    pattern = re.compile(r'\b(' + r'|'.join(stopwords_list) + r')\b\s*')
    sentence = pattern.sub('', sentence)

    #print(sentence)
    return sentence

def get_twitter_data(tag):
    
    dir_name = create_data_directory(tag)
    
    client = tweepy.Client(
    bearer_token="AAAAAAAAAAAAAAAAAAAAAOz8nwEAAAAA74cGh1%2FFtKsIYeK12bPZBEwFigk%3DO0MQomSDOEhii2KgYZzHu7YxgowR1andUW9BZfyMmoOEdqxQuy")
    query = f"#{tag} -is:retweet lang:en"
    paginator = tweepy.Paginator(
        client.search_recent_tweets,
        query=query,
        max_results=100,
        limit=5
    )

    tweet_list = [tweet for tweet in paginator.flatten()]

    tweet_list_df = pd.DataFrame(tweet_list)
    tweet_list_df = pd.DataFrame(tweet_list_df['text'])

    cleaned_tweets = [preprocess_text(tweet) for tweet in tweet_list_df['text']]

    tweet_list_df['cleaned'] = pd.DataFrame(cleaned_tweets)
    
    filename = f"Tweets.csv"
    
    # add the directory name to the filename
    file_path = os.path.join(dir_name, filename)
    
    tweet_list_df.to_csv(file_path, index=False)

    return tweet_list_df, dir_name

def get_google_data(searchword, start_date, end_date):
    
    dir_name = create_data_directory(searchword)
    
    api = NewsApiClient(api_key='71262aa695c24084be3f41c65507b5f2')
    
    all_search = api.get_everything(q=searchword,
                            from_param=start_date,
                            to=end_date,
                            language='en')
    article_data = {
    "title": [article["title"] for article in all_search["articles"]],
    "description": [article["description"] for article in all_search["articles"]],
    "content": [article["content"] for article in all_search["articles"]]
    }
    
    google_list_df = pd.DataFrame(article_data)

    cleaned_title = [preprocess_text(title) for title in google_list_df['title']]
    cleaned_decription = [preprocess_text(description) for description in google_list_df['description']]
    cleaned_content = [preprocess_text(content) for content in google_list_df['content']]
        
    google_list_df['cleaned_title'] = pd.DataFrame(cleaned_title)
    google_list_df['cleaned_description'] = pd.DataFrame(cleaned_decription)
    google_list_df['cleaned_content'] = pd.DataFrame(cleaned_content)
    
    google_list_df['cleaned'] = google_list_df['cleaned_title'] + google_list_df['cleaned_description'] + google_list_df['cleaned_content']
    
    filename = f"GoogleNewsArticles.csv"
    
    # add the directory name to the filename
    file_path = os.path.join(dir_name, filename)
    
    google_list_df.to_csv(file_path, index=False)
    
    return google_list_df, dir_name