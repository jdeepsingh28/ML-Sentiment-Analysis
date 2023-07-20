from transformers import pipeline
import pandas as pd

def apply_model(data, data_source):
    if data_source == "twitter":
        nlp = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

        data['sentiment'] = data['cleaned'].apply(lambda x: nlp(x)[0]['label'])

        total = len(data)
        positive = len(data[data['sentiment'] == "Positive"])
        negative = len(data[data['sentiment'] == "Negative"])
        neutral = len(data[data['sentiment'] == "Neutral"])
        positive_percentage = (positive / total) * 100
        negative_percentage = (negative / total) * 100
        
        print("Positive Percentage: " + str(positive_percentage))
        print("Negative Percentage: " + str(negative_percentage))
        
    elif data_source == "google news":
        nlp = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
        
        data['sentiment'] = data['cleaned'].apply(lambda x: nlp(x)[0]['label'])
        
        total = len(data)
        positive = len(data[data['sentiment'] == "POSITIVE"])
        negative = len(data[data['sentiment'] == "NEGATIVE"])
        positive_percentage = (positive / total) * 100
        negative_percentage = (negative / total) * 100
        
        print("Positive Percentage: " + str(positive_percentage))
        print("Negative Percentage: " + str(negative_percentage))
    
    else:
        raise ValueError("Invalid Source (Valid Sources: twitter or googlenews)")
        
        

    
    