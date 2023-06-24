import pandas as pd
from transformers import pipeline, AutoTokenizer, TFAutoModelWithLMHead
import os
from joblib import Parallel, delayed

nlp = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def count_values_in_column(data,feature):
    total = data.loc[:, feature].value_counts(dropna=False)
    percentage = round(data.loc[:, feature].value_counts(dropna=False,normalize=True)*100,2)
    return pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage'])

def process_batch(i, texts, batch_size):
    batch = texts[i:i+batch_size]
    scores = []
    try:
        results = nlp(batch)
    except Exception as e:
        print(f"Error in NLP processing: {e}")
        return scores  # Return empty list for this batch

    for j, result in enumerate(results):
        index = i + j
        try:
            pos_threshold = 0.6  # set your own threshold
            neg_threshold = 0.6  # set your own threshold

            if result['label'] == 'LABEL_1' and result['score'] > pos_threshold:
                sentiment = 'positive'
                pos = result['score']
                neg = 0
                neu = 0
            elif result['label'] == 'LABEL_0' and result['score'] > neg_threshold:
                sentiment = 'negative'
                pos = 0
                neg = result['score']
                neu = 0
            else:
                sentiment = 'neutral'
                pos = 0
                neg = 0
                neu = 1

            scores.append(
                {'index': index, 'neg': neg, 'neu': neu, 'pos': pos, 'sentiment': sentiment})
        except Exception as e:
            print(f"Error in processing results: {e}")
            continue  # Skip this result
    return scores

def RoBERTa_model(data, dir_name):
    #choose batch size based on input size
    if len(data) > 10000:
        batch_size = 200
    elif len(data) > 5000:
        batch_size = 100
    else:
        batch_size = 16
    
    # prepare the texts as a list
    try:
        texts = data['cleaned'].tolist()
    except Exception as e:
        print(f"Error in converting DataFrame to list: {e}")
        return

    # create empty list to hold scores
    scores = []

    # set filename as "Sentiment-" followed by the current date and time
    filename = f"Sentiment.txt"
    # add the directory name to the filename
    file_path = os.path.join(dir_name, filename)

    # process texts in batches
    for i in range(0, len(texts), batch_size):
        batch_scores = process_batch(i, texts, batch_size)
        scores.extend(batch_scores)

        # create scores_df for this batch
        scores_df = pd.DataFrame(scores).set_index('index')

        # join the original DataFrame with the scores DataFrame
        data_batch = data.loc[scores_df.index].join(scores_df)

        # process sentiment counts and write to file for this batch
        try:
            data_negative = data_batch[data_batch['sentiment'] == 'negative']
            data_positive = data_batch[data_batch['sentiment'] == 'positive']
            data_neutral = data_batch[data_batch['sentiment'] == 'neutral']

            sentiment_counts = count_values_in_column(data_batch, 'sentiment')

            with open(file_path, 'a') as f:
                f.write(f"\nBatch {i//batch_size + 1} Sentiment Counts:\n")
                f.write(sentiment_counts.to_string())
                f.write("\n")
        except Exception as e:
            print(f"Error in DataFrame operations or file writing: {e}")
            return

    # process sentiment counts and write to file after all batches have been processed
    try:
        scores_df = pd.DataFrame(scores).set_index('index')
        data = data.join(scores_df)

        data_negative = data[data['sentiment'] == 'negative']
        data_positive = data[data['sentiment'] == 'positive']
        data_neutral = data[data['sentiment'] == 'neutral']

        sentiment_counts = count_values_in_column(data, 'sentiment')

        print("Final Sentiment Counts:")
        print(sentiment_counts)

        with open(file_path, 'a') as f:
            f.write("\nFinal Sentiment Counts:\n")
            f.write(sentiment_counts.to_string())
            f.write("\n")
    except Exception as e:
        print(f"Error in DataFrame operations or file writing: {e}")



