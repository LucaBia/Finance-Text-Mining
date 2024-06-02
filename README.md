<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />
<img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/Visual_Studio_Code-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white" />
<img src="https://forthebadge.com/images/badges/powered-by-coffee.svg" style="height:28px;" />




# Financial News Sentiment Analysis and Stock Price Prediction

This project aims to determine if training a text classifier on financial news headlines can explain (after-the-fact) the price movements on the stock market on that day. We utilize various machine learning models, including traditional models and a Hugging Face transformer-based model, to classify the sentiment of financial news and analyze its impact on stock prices.


## Requirements

- Python 3.7+
- Jupyter Notebook
- Required Python libraries:
  - numpy
  - pandas
  - eodhd
  - textblob
  - nltk
  - sklearn
  - scipy
  - yfinance
  - xgboost
  - torch
  - transformers
  - matplotlib
  - seaborn
  - imblearn
  - optuna

You can install the necessary libraries using pip:
```bash
pip3 install numpy pandas eodhd textblob nltk sklearn scipy yfinance xgboost torch transformers matplotlib seaborn imblearn optuna
```

## Data Collection

### Financial News

Financial news headlines are collected using the EODHD API’s Financial News endpoint. The data is cleaned and preprocessed to remove unnecessary characters while preserving numbers and percentage symbols, which are important for financial news analysis.

### Stock Prices

Stock prices are collected using the yfinance library. The price changes are calculated using percentage change to normalize the data across different stock symbols.

### Sentiment Analysis

We perform sentiment analysis on the news headlines using two methods:
1.	TextBlob: Provides a simple API for diving into common natural language processing (NLP) tasks.
2.	VADER (Valence Aware Dictionary and sEntiment Reasoner): A lexicon and rule-based sentiment analysis tool specifically attuned to sentiments expressed in social media.

Both sentiment scores are included as features for further analysis.

## Machine Learning Models

### Traditional Models

We use several traditional machine learning models to classify the sentiment and predict the stock price movements:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- XGBoost

These models are trained and evaluated on the preprocessed data. The performance is measured using accuracy and classification reports.

### Hugging Face Transformer Model

We utilize the DistilBERT model from Hugging Face to classify the sentiment of news headlines. The model is fine-tuned on our dataset, and hyperparameters are optimized using Optuna. The balanced dataset is created using RandomOverSampler to handle class imbalance.

### Evaluation

The models are evaluated on both validation and test datasets. We compare the performance of the traditional models with the Hugging Face transformer model to determine the best approach for sentiment analysis and stock price prediction.