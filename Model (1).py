from transformers import pipeline
import pandas as pd
import numpy as np

nlp = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

def apply_model(data, batch_size=16): 
    data['sentiment'] = data['cleaned'].apply(lambda x: nlp(x)[0]['label'])
        
    total = len(data)
    positive = len(data[data['sentiment'] == "positive"])
    negative = len(data[data['sentiment'] == "negative"])
    neutral = len(data[data['sentiment'] == "neutral"])
    positive_percentage = (positive / total) * 100
    negative_percentage = (negative / total) * 100
    neutral_percentage = (neutral / total) * 100

    return {
        "Positive Percentage": positive_percentage,
        "Negative Percentage": negative_percentage,
        "Neutral Percentage": neutral_percentage
    }