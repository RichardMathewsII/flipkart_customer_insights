from gpt_utils import send_prompt, set_api_key
from lda import extract_topics_from_reviews
import pandas as pd
import streamlit as st
import json


st.cache_data()
def generate_customer_insights_report(df: pd.DataFrame, product: str, status) -> str:
    '''The main script that runs topic modeling, sends prompts to GPT, and creates
    a markdown report detailing the results of customer reviews analysis to sellers

    Parameters
    ----------
    df : a dataframe with products and customer reviews
    product : the product to generate a customer insights report for
    status : a streamlit object to print status updates to in the app

    Returns
    -------
    a markdown report detailing product description, customer praise, customer
    complaints, product improvement ideas, and a conclusion
    '''

    # check if report has been cached
    # saves costs and compute to reuse previously generated reports
    with open('reports.json') as f:
        try:
            reports = json.load(f)
        except:
            # json file not yet created
            reports = {}
    if product in reports.keys():
        # report has been previously run
        # return cached report
        return reports[product]

    
    set_api_key()  # pass API key to openAI client
    total_steps = 4  # number of steps in status printing

    # Step 1: extract positive and negative topics from customer reviews
    status.write(f"Topic modeling [1/{total_steps}]")
    negative_topics = extract_topics_from_reviews(dataset=df, product=product, sentiment="negative")
    positive_topics = extract_topics_from_reviews(dataset=df, product=product, sentiment="positive")
    negative_keywords = ', '.join(negative_topics.Topic.to_list())
    positive_keywords = ', '.join(positive_topics.Topic.to_list())
    prompt="What product does this description describe: "+product  # get product description
    product_description = send_prompt(prompt=prompt)

    # Step 2: summarize customer complaints and praise from topic modeling results
    status.write(f"Processing reviews [2/{total_steps}]")
    prompt="What type of object is this product description describing:"+product_description+" Answer in the format: 'Title: {title}'"
    obj = send_prompt(prompt)

    prompt=f"Summarize the problems customers are having with {obj.replace('Title: ', '')} based on these keywords extracted from product reviews: {negative_keywords}"
    neg_reviews_summary = send_prompt(prompt)

    prompt=f"Summarize what customers like about {obj.replace('Title: ', '')} based on these keywords extracted from product reviews: {positive_keywords}"
    pos_reviews_summary = send_prompt(prompt)

    # Step 3: brainstorm product improvement ideas based on summary of customer complaints
    status.write(f"Brainstorming improvement ideas [3/{total_steps}]")
    prompt=f"Brainstorm some ways a seller could improve their product {obj} based on the following summary of customer reviews: {neg_reviews_summary}"
    improvements = send_prompt(prompt)

    # Step 4: generate a markdown report for the seller detailing the results of the customer analysis
    status.write(f"Generating report [4/{total_steps}]")
    prompt=f"Generate a markdown report focused on improving {obj} with sections for product description, customer praise, customer complaints, potential improvements, and conclusion from the following information: \
         {obj}\n   \
        {product_description}\n \
            {pos_reviews_summary}\n \
            {neg_reviews_summary}\n \
                {improvements}\n "
    report = send_prompt(prompt)

    # cache the report
    reports[product] = report
    with open('reports.json', 'w') as f:
        json.dump(reports, f)
    
    return report
