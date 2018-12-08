from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

vectorizer = CountVectorizer()

news_data = pd.read_csv("/Users/jonathanhale/Documents/Courses/Machine Learning/example_data/news_sample (1).csv")



def bag_of_words (news_data):
    """bag_of_words function to process the news 'headline' data."""
    corpus = news_data['headline'].values.tolist()
    mat_corpus = vectorizer.fit_transform(corpus).toarray()
    news_data['headline'] = list(mat_corpus)
    return news_data

bag_of_words(news_data)


