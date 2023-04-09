# Analyzing Flipkart Reviews for Product Improvement
> This repo houses the code for a hypothetical AI system that delivers product insights to Flipkart sellers, including key patterns in customer complaints and praise, as well as recommendations to improve the product.

## Dataset
TODO

## Approach
The general approach is to identify the dominant themes in positive and negative customer reviews for each product and feed the customer complaint patterns into GPT to produce product improvement recommendations. The Latent Dirichlet Allocation (LDA) algorithm is an effective technique for extracting out "topics" from a corpus. I assume negative customer reviews can be distilled into a small set of common complaints that a topic modeling algorithm like LDA can find, and this reduced representation of the corpus can be fed to an LLM model to brainstorm product improvement ideas.


![](app/assets/workflow_I.png)

![](app/assets/workflow_II.png)

**Why do topic modeling rather than sending GPT all the reviews?**

One might wonder why we can't just send GPT all the reviews and ask it to ideate product improvements? Technically, we can, but data science must consider the costs of scaling such a solution. OpenAI charges on a per-token basis, so sending all reviews would be orders of magnitude more expensive. We can get similar results by collapsing an entire corpus of product reviews into a fixed number of tokens by applying topic modeling beforehand. 

For example, in this application, the number of tokens sent to GPT is just 50 tokens (5 topics and 5 tokens per topic for both positive and negative reviews). In total, there are 8,101,246 tokens in this corpus of product reviews. By applying LDA (10 topics per product, 5 tokens per topic), I reduce this token count to 15,825 tokens! That is a **99.8% reduction in costs**. That's not much for this dataset, but it will be after scaling to all sellers and products on Flipkart, not to mention the added latency of sending GPT all of those tokens.

![](app/assets/cost_reduction.png)

This is an important demonstration of how data scientists should think about scaling solutions involving LLMs. It's not wise to brute force NLP problems with LLMs when basic preprocessing and token reduction methods prior to the LLM prompts can slash most of the costs and latency in a scaled solution.


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

### Demo
[![Watch Demo](https://img.youtube.com/vi/SjW10t-bOO8/0.jpg)](https://youtu.be/SjW10t-bOO8)

