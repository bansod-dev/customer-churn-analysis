# Customer Churn Analysis & Prediction

A Machine Learning project that analyzes customer behavior and predicts whether a customer is likely to churn using a Random Forest classifier.

## 🚀 Live Demo

**Streamlit App:** https://customer-churn-analysis-gkpfhdqhbndmzboak5z6xq.streamlit.app/

## 📌 Project Overview

Customer churn is a major challenge for subscription-based businesses. This project uses the Telco Customer Churn dataset to analyze customer characteristics and build a machine learning model that predicts customer churn.

The trained model is integrated into a Streamlit web application where users can enter customer details and receive a churn prediction with probability.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset.

The dataset contains customer information such as:

* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming Services
* Contract
* Payment Method
* Monthly Charges
* Total Charges
* Churn

## 🔄 Machine Learning Workflow

1. Data loading
2. Data cleaning
3. Handling categorical variables
4. Feature encoding
5. Feature scaling
6. Train-test split
7. Model training
8. Model evaluation
9. Model serialization using Joblib
10. Streamlit deployment

## 🤖 Model

### Random Forest Classifier

The Random Forest model was selected as the final model for the application.

The model uses customer demographic, service, contract, and billing information to predict churn.

### Model Performance

Test-set results:

| Metric           | Score |
| ---------------- | ----: |
| Accuracy         |  ~78% |
| Class 0 F1-score |  0.84 |
| Class 1 F1-score |  0.61 |

Confusion Matrix:

```text
[[851, 185],
 [128, 245]]
```

The model performs differently across the two classes, with lower recall and F1-score for the churn class. This is an important consideration because identifying customers who are likely to churn is the main objective of the project.

## 🖥️ Streamlit Application

The Streamlit application allows users to enter customer information including:

* Customer demographics
* Tenure
* Monthly charges
* Total charges
* Internet services
* Security services
* Streaming services
* Contract type
* Payment method

The application then returns:

* Churn prediction
* Probability of churn

## 📁 Project Structure

```text
customer-churn-analysis/
│
├── app.py
├── churn_random_forest.pkl
├── churn_scaler.pkl
├── churn_columns.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Run Locally

Clone the repository:

```bash
git clone https://github.com/bansod-dev/customer-churn-analysis.git
```

Navigate to the project directory:

```bash
cd customer-churn-analysis
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 🔮 Future Improvements

* Add Explainable AI using SHAP
* Improve minority-class detection
* Experiment with additional ML models
* Add interactive data visualizations
* Improve model performance through hyperparameter tuning

## 👨‍💻 Author

**Ghanshyam Bansod**

AI & Data Science Student | Aspiring Data Scientist

GitHub: https://github.com/bansod-dev
