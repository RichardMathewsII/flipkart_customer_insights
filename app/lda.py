import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from typing import List

def extract_topics_from_reviews(
        dataset: pd.DataFrame, 
        product: str, 
        sentiment: str = 'negative', 
        num_topics: int = 5, 
        num_words_per_topic: int = 5
    ) -> pd.DataFrame:
    '''Extract subset of data for given product and sentiment, and feed this
    subset to a topic modeling algorithm (LDA).

    Parameters
    ----------
    dataset : dataframe with product names and customer reviews
    product : the product name to conduct topic modeling on
    sentiment (optional) : the sentiment of reviews to model ('positive' or 'negative'), defaults to 'negative'
    num_topics (optional) : the number of review topics to extract, defaults to 5
    num_words_per_topic (optional) : the number of tokens to represent each topic, defaults to 5

    Returns
    -------
    a dataframe with the topics ranked by importance
    '''
    # filter for requested sentiment and group on products    
    by_product = dataset.loc[dataset.sentiment == sentiment, :].groupby(by=["ProductName"], group_keys=False)
    # select the subset of reviews for the requested product
    product_group = by_product.get_group(name=product)
    # extract dominant patterns in the reviews for the product
    return _apply_topic_modeling(group=product_group, num_topics=num_topics, num_words_per_topic=num_words_per_topic)


def _apply_topic_modeling(group: pd.DataFrame, num_topics: int = 5, num_words_per_topic: int = 5) -> pd.DataFrame:
    '''Run topic modeling on product reviews and structure result in a dataframe

    Parameters
    ----------
    group : Pandas group of reviews for a particular product
    num_topics (optional) : number of topics to extract, defaults to 5
    num_words_per_topic (optional) : number of words per topic, defaults to 5

    Returns
    -------
    dataframe with ranked topics
    '''    
    product = group.ProductName.values[0]  # get name of product
    # extract topics
    topics = _find_topics(group.Summary, num_topics=num_topics, num_words_per_topic=num_words_per_topic)
    # structure output
    return pd.DataFrame(
        data={
        "Product": [product for _ in range(num_topics)], 
        "Topic Number": list(range(1, num_topics+1)), 
        "Topic": topics
        })


def _find_topics(reviews: pd.Series, num_topics: int = 5, num_words_per_topic: int = 5) -> List[str]:
    '''Applied LDA to extract topics from set of product reviews

    Parameters
    ----------
    reviews : product reviews
    num_topics (optional) : number of topics to extract, defaults to 5
    num_words_per_topic (optional) : number of words in each topic, defaults to 5

    Returns
    -------
    List of topics
    '''
    # convert set of reviews to bag of words
    # vocab = CountVectorizer(stop_words='english', max_df=.1, max_features=10000)
    vocab = TfidfVectorizer(stop_words='english', max_df=0.5, max_features=100000)
    bag_of_words = vocab.fit_transform(reviews)

    # apply lda to get topics
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=123, learning_method='batch')
    lda.fit(bag_of_words)

    # structure topics in list
    feature_names = vocab.get_feature_names_out()
    topics = []
    for _, topic in enumerate(lda.components_): 
        topics.append(" ".join([feature_names[i] for i in topic.argsort()[:-num_words_per_topic - 1:-1]]))
    return topics