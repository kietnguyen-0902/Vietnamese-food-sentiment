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
```
## ⚙️ Tech Stack
*   **Data Retrieval:** `kagglehub` (Automated dataset fetching)
*   **Data Manipulation & Analysis:** `pandas`, `numpy`, `scipy`
*   **NLP & Text Processing:** `pyvi` (Vietnamese Tokenizer)
*   **Feature Engineering:** `scikit-learn` (TF-IDF, StandardScaler)
*   **Visualization:** `matplotlib`, `seaborn`

---

## ⚖️ Model Trade-offs & Business Use Cases

The dataset exhibits a class imbalance (~3:1 ratio for Positive:Negative). Both custom models perform exceptionally well on the TF-IDF feature space, but their structural differences make them suitable for different business scenarios:

### 1. Naive Bayes (F1-Score: ~0.89)
*   **Scenario:** General sentiment tracking, Social Listening, and Dashboarding.
*   **Why:** NB achieved an outstanding **Recall (0.98)** for positive reviews. It aggressively identifies positive sentiment, ensuring no good review slips through the cracks. It is also exceptionally fast to train and infer, making it ideal for real-time streaming data or resource-constrained environments.

### 2. Support Vector Machine (F1-Score: ~0.87)
*   **Scenario:** Automated Customer Reward Systems or Critical Alerting.
*   **Why:** SVM offers a higher **Precision (0.94)** for positive reviews. It is more "cautious" before labeling a comment as positive. If a restaurant's system automatically issues a discount voucher for a 5-star text review, SVM minimizes *False Positives* (e.g., misclassifying a sarcastic negative review as positive), ultimately saving costs for the business.

---

## 🚀 How to Run

**Note on Dataset:** The dataset (`.csv`) is intentionally excluded from this repository to adhere to best practices for source control. 

The notebook is fully automated. To run the project:
1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3.Open notebooks/vsa_food_reviews.ipynb via Jupyter Notebook or Google Colab.
4.Run all cells. The script will automatically connect to Kaggle via kagglehub, download the dataset into your local memory, and execute the entire pipeline from EDA to Model Inference.

##💡 Inference Demo
The pipeline includes a production-ready inference function that applies the exact preprocessing/vectorization steps to raw user text before prediction.

Text: 'Tuyệt vời! Đồ ăn tươi, nóng, ngon. Nhân viên thân thiện.'
Prediction: Positive 😊
--------------------------------------------------
Text: 'Chất lượng tệ, giá đắt, service chậm. Không bao giờ quay lại.'
Prediction: Negative 😞
