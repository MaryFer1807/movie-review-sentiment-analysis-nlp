# Movie Review Sentiment Analysis – NLP Classification Project

## Project Overview
This project focuses on building a **sentiment analysis system** for movie reviews. The goal is to automatically classify reviews as **positive or negative** using machine learning and natural language processing techniques.

The dataset is based on IMDB movie reviews and reflects a real-world text classification problem where model quality is evaluated using the **F1 score**, with a required threshold of **0.85**.

## Objective
The main objectives of this project are to:
- Load and explore labeled movie review data.
- Analyze class balance and textual characteristics.
- Preprocess raw text data for machine learning.
- Train and compare multiple text classification models.
- Evaluate model performance using the F1 metric.
- Classify custom-written reviews and analyze model behavior.
- Explain differences between automated evaluation and manual predictions.

## Dataset
The dataset used:
- `imdb_reviews.tsv`

Key fields:
- `review` — text of the movie review
- `pos` — target label (0 = negative, 1 = positive)
- `ds_part` — dataset split (`train` / `test`)

The dataset was originally published by Maas et al. (2011) for sentiment analysis research.

## Exploratory Data Analysis
- Examined the distribution of positive and negative reviews.
- Identified whether class imbalance is present.
- Analyzed review lengths and text characteristics.
- Drew conclusions about the suitability of the dataset for classification.

## Text Preprocessing
- Converted text to lowercase.
- Removed punctuation and unnecessary symbols.
- Tokenized text data.
- Vectorized text using techniques such as:
  - Bag of Words
  - TF-IDF
- Ensured consistent preprocessing across training and test datasets.

## Modeling Approach
At least three different models were trained and evaluated, including:
- Logistic Regression
- Gradient Boosting–based classifiers
- Additional baseline or tree-based models for comparison

Each model was trained on the training set and evaluated on the test set using the **F1 score**.

## Model Evaluation
- Compared F1 scores across all trained models.
- Verified that the best-performing model achieved **F1 ≥ 0.85**.
- Used a unified evaluation routine to ensure fair comparison.
- Analyzed precision–recall trade-offs.

## Custom Review Classification
- Wrote custom movie reviews manually.
- Classified these reviews using all trained models.
- Compared predictions with expected sentiment.
- Identified differences between automated metrics and real-world text interpretation.
- Explained why models may behave differently on short or ambiguous reviews.

## Results and Findings
- Demonstrated that proper text preprocessing significantly improves model performance.
- Showed that linear models perform strongly for sentiment analysis tasks.
- Highlighted strengths and weaknesses of different classifiers.
- Confirmed that the project requirements were fully met.

## Tools Used
- Python
- Pandas
- NumPy
- Scikit-learn
- NLP vectorization techniques
- Matplotlib
- Jupyter Notebook

## Business Value
This project demonstrates the ability to:
- Work with unstructured text data.
- Build and evaluate NLP classification models.
- Apply appropriate metrics for imbalanced or semantic tasks.
- Translate machine learning outputs into explainable insights.

The approach can be applied to content moderation, review filtering, recommendation systems, and customer feedback analysis.
