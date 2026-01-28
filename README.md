# Ensemble_learning
https://stacking-loan-approval.streamlit.app/

# 🎯 Smart Loan Approval System – Stacking Model

## 📌 Project Overview
This repository contains a **Smart Loan Approval System** built using a **Stacking Ensemble Machine Learning** approach.  
The goal is to predict whether a loan will be approved by a customer using combined predictions from multiple models for better accuracy and stability.

The app is implemented in **Python** using **Streamlit** for the frontend and **scikit-learn** for modeling.

---

## 🚀 Features

- 👩‍💻 **Stacking Ensemble Model**
  - Base Models: Logistic Regression, Decision Tree, Random Forest
  - Meta Model: Logistic Regression
- 📊 Interactive input form for applicant details
- 🧠 Displays base model predictions and final stacking decision
- 💡 Confidence score and business explanation for predictions
- 🎨 Responsive UI with custom styling

---

## 📂 Repository Contents

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application for loan prediction |
| `requirements.txt` | Python dependencies required |
| `loan.csv` | Dataset used for training the model |
| `boosting.ipynb` | Notebook on boosting approaches |
| `README.md` | Project documentation (this file) |

---

## 🧰 Technology Stack

- Python
- scikit-learn for Machine Learning
- Streamlit for Web App UI
- pandas, numpy for data handling

---

## 🛠 Installation

1. **Clone the repository**
```bash
git clone https://github.com/Khushibung05/Ensemble_learning.git
cd Ensemble_learning

2. **Install dependencies**
```bash
pip install -r requirements.txt
Run the Streamlit app

## 📈 How to Use
Enter applicant information in the sidebar:

Applicant Income

Co-Applicant Income

Loan Amount

Loan Amount Term

Credit History

Employment Status

Property Area

Click “Check Loan Eligibility (Stacking Model)”

## View:

Base model predictions

Final stacking decision

Confidence score

## Business explanation

🙌 Business Impact
This system helps lenders evaluate loan applications with improved accuracy by combining different models. It provides interpretability and stability for financial decision-making.

