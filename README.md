# 📊 Vietnamese Sentiment Analysis (VSA): Food Reviews

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Custom%20Models-orange)
![NLP](https://img.shields.io/badge/NLP-Vietnamese-success)

## 🎯 Project Overview
This project focuses on building a robust Machine Learning pipeline for Vietnamese Sentiment Analysis (VSA) applied to food and restaurant reviews. The primary objective is to classify customer feedback into two distinct sentiments:
*   **Positive (1):** Satisfied customer experience.
*   **Negative (0):** Dissatisfied customer experience.

**Key Highlight:** Rather than relying solely on high-level library abstractions like `scikit-learn` for modeling, the core classification algorithms (**Multinomial Naive Bayes** and **Support Vector Machine**) are **implemented entirely from scratch**. This demonstrates a deep mathematical understanding of the underlying mechanics, including Laplace smoothing for Naive Bayes and a custom Sequential Minimal Optimization (SMO) algorithm for the SVM.

---

## 📂 Repository Structure
```text
vietnamese-food-sentiment/
│
├── models/
│   ├── tfidf_vectorizer.pkl        # Pre-trained TF-IDF vectorizer (5000 features)
│   └── length_scaler.pkl           # Pre-trained length scaler 
│
├── notebooks/
│   └── vsa_food_reviews.ipynb      # Main execution pipeline & experiments
│
├── .gitignore                      # Excludes heavy data and cache files
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
