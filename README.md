<div align="center">
  
  # 🌳 Decision Tree Classifier for Bank Marketing Prediction

  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  
  **SkillCraft Technology — Data Science Internship | Task 3 (SCT_DS_3)**
  
  *A machine learning project focused on predictive modeling to optimize direct marketing campaigns for banking institutions.*
  
</div>

---

## 📖 Task Definition & Business Context

In the highly competitive banking sector, direct marketing campaigns (like telemarketing) remain a vital strategy for customer acquisition. However, indiscriminately calling customers is expensive, inefficient, and can lead to customer fatigue.

**The Objective:** 
Build a predictive model using a **Decision Tree Classifier** that determines whether a customer will subscribe to a term deposit (a financial product) based on their demographic information and behavioral history. 

By accurately predicting which customers are most likely to convert, the bank can:
- **Optimize Resource Allocation:** Focus sales teams on high-probability targets.
- **Reduce Marketing Costs:** Minimize the number of unsuccessful calls.
- **Enhance Customer Experience:** Avoid bothering customers who are unlikely to be interested.

---

## 📊 Dataset Overview

We utilize the renowned [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing), consisting of data collected from real-world direct marketing campaigns conducted by a Portuguese banking institution.

| Metric | Details |
| :--- | :--- |
| **Total Records** | 4,521 distinct customer interactions |
| **Input Features** | 16 features spanning demographics, financial status, and campaign history |
| **Target Variable** | `y` — Did the client subscribe to a term deposit? (`yes` or `no`) |

### Key Feature Categories Analyzed:
1. **Demographics:** Age, Job type, Marital status, Education level.
2. **Financial Health:** Default history, Average yearly balance, Housing loan status, Personal loan status.
3. **Current Campaign Interaction:** Contact type, Contact day/month, Duration of the call.
4. **Historical Interaction:** Previous campaign outcomes, Days since last contact.

---

## 🧠 The Machine Learning Pipeline

Our implementation follows a rigorous, production-ready machine learning pipeline designed for clarity, reproducibility, and high performance.

### 1. Data Preprocessing & Encoding
Decision trees require numerical input. We applied **One-Hot Encoding** (via `pandas.get_dummies()`) to transform all 9 categorical variables into 51 distinct binary features, ensuring the algorithm can mathematically process all demographic signals without assuming any ordinal relationships.

### 2. Strategic Data Splitting
To evaluate the model's true predictive power on unseen data, the dataset was split into:
- **Training Set (70%):** 3,164 samples used to train the model.
- **Testing Set (30%):** 1,357 samples reserved strictly for unbiased evaluation.

### 3. Model Training: Why a Decision Tree?
A `DecisionTreeClassifier` was selected primarily for its **high interpretability**. In business contexts like banking, stakeholders need to understand *why* a model makes a specific prediction. 
* We constrained the tree with `max_depth=3` to prevent overfitting and ensure the resulting decision rules remain human-readable and actionable for the marketing team.

### 4. Evaluation & Insights
The model achieved an impressive **89.68% overall accuracy**. More importantly, analyzing the *Feature Importances* revealed exactly what drives customer conversion:
1. **Call Duration (57.25%):** Longer conversations heavily correlate with a successful sale.
2. **Previous Success (32.36%):** Customers who bought products in past campaigns are highly likely to buy again.
3. **Contact Timing (8.23%):** Reaching out during specific months (like October) yields higher conversion rates.

---

## 🚀 Getting Started

Want to run this predictive model on your local machine? Follow these simple steps:

### Prerequisites
Ensure you have Python installed, then install the required libraries:
```bash
pip install pandas scikit-learn matplotlib
```

### Execution
Run the main script from your terminal:
```bash
python task3.py
```

### Expected Output
1. A detailed **Classification Report** printed directly to your console, detailing Precision, Recall, and F1-Scores.
2. An elegant, dark-themed visualization of the generated tree saved as `decision_tree_model.png`.

---

## 📂 Repository Structure

```text
SC_DS_3/
├── task3.py                  # The core ML pipeline script
├── bank.csv                  # The source dataset
├── decision_tree_model.png   # Auto-generated visualization of the model
├── requirements.txt          # Environment dependencies
└── README.md                 # Project documentation (You are here)
```

---

<div align="center">
  
  ### 👨‍💻 Developed by **Jinendra Banthia**
  *Data Science Intern @ SkillCraft Technology*
  
  <p><i>Completed as part of the SkillCraft Technology Data Science Internship Program.</i></p>

</div>
