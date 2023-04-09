# Analyzing Flipkart Reviews for Product Improvement
> This repo houses the code for a hypothetical AI system that delivers product insights to Flipkart sellers, including key patterns in customer complaints and praise, as well as recommendations to improve the product.

## Dataset
TODO

## Notebooks
The notebooks demonstrate EDA, cleaning steps, and NLP work.
- [cleaning.ipynb](cleaning.ipynb) - data cleaning
- [topic_modeling.ipynb](topic_modeling.ipynb) - topic modeling and GPT example

## App
The `app/` directory contains code to run the data cleaning, topic modeling, GPT prompt requests, and streamlit app.
- [main](app/main.py) - code to run end-to-end analysis on a given product and generate customer insights report
- [cleaning](app/cleaning.py) - code to clean Flipkart product reviews
- [lda](app/lda.py) - code to run LDA topic modeling
- [gpt_utils](app/gpt_utils.py) - code to communicate with OpenAI GPT models