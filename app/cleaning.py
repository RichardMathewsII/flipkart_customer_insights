import pandas as pd
import streamlit as st


@st.cache_data()
def clean_dataset(dataset: pd.DataFrame):
    """
    Cleans a dataset by removing any rows with missing values.
    
    Args:
    - dataset (pandas.DataFrame): The dataset to clean.
    
    Returns:
    - A pandas DataFrame containing the cleaned dataset.
    """
    dataset.dropna(inplace=True)
    dataset = dataset[dataset.Rate.apply(_is_valid_rating)]
    dataset.ProductName = dataset.ProductName.astype(str)
    dataset.Review = dataset.Review.astype(str)
    dataset.Summary = dataset.Summary.astype(str)
    dataset.Rate = dataset.Rate.astype(int)
    dataset = _label_sentiment(dataset)
    # keep products with at least 30 reviews
    dataset = _filter_min_num_reviews(dataset, min_reviews=30)
    return dataset


def _label_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    '''Add column sentiment to dataframe with 'neutral', 
    'positive', or 'negative'. Assigned based on rating.

    Parameters
    ----------
    df : reviews dataset

    Returns
    -------
    dataset with a column for sentiment
    '''
    df.loc[:, 'sentiment'] = 'neutral'
    df.loc[df.Rate < 3, 'sentiment'] = 'negative'
    df.loc[df.Rate > 3, 'sentiment'] = 'positive'
    return df


def _filter_min_num_reviews(df: pd.DataFrame, min_reviews: int) -> pd.DataFrame:
    review_counts = df.ProductName.value_counts()
    products_to_keep = review_counts[review_counts >= min_reviews].index.to_list()
    return df.loc[df.ProductName.isin(products_to_keep), :]


def _is_valid_rating(rate) -> bool:
    '''Determine if rating is valid value'''
    try:
        # valid: can be cast as int
        a = int(rate)
        return True
    except:
        # could not be cast as int, thus invalid
        return False
    
