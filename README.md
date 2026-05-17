# 🌳 Decision Tree Classifier — Bank Term Deposit Prediction

> **SkillCraft Technology — Data Science Internship | Task 3 (SCT_DS_3)**

A supervised machine learning project that builds a **Decision Tree Classifier** to predict whether a bank client will subscribe to a term deposit, using the [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

---

## 📋 Problem Statement

Build a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data. The dataset originates from direct marketing campaigns (phone calls) of a Portuguese banking institution.

---

## 📂 Dataset Overview

| Property            | Value                                     |
|---------------------|-------------------------------------------|
| **Source**           | UCI Machine Learning Repository            |
| **File**             | `bank.csv` (semicolon-separated)           |
| **Instances**        | 4,521                                      |
| **Features**         | 16 (Categorical + Integer)                 |
| **Target Variable**  | `y` — Has the client subscribed? (yes/no)  |
| **Missing Values**   | None                                       |

### Feature Categories

| Category                  | Features                                                          |
|---------------------------|-------------------------------------------------------------------|
| **Client Demographics**   | `age`, `job`, `marital`, `education`, `default`, `balance`         |
| **Loan Information**      | `housing`, `loan`                                                  |
| **Campaign Details**      | `contact`, `day`, `month`, `duration`, `campaign`                  |
| **Previous Campaign**     | `pdays`, `previous`, `poutcome`                                    |

---

## 🔧 ML Pipeline

The project follows a structured machine learning pipeline:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  1. ENCODE   │───▶│   2. SPLIT   │───▶│   3. TRAIN   │───▶│ 4. EVALUATE  │
│              │    │              │    │              │    │              │
│ One-Hot      │    │ 70% Train    │    │ DecisionTree │    │ Accuracy     │
│ Encoding     │    │ 30% Test     │    │ max_depth=3  │    │ Report       │
│ (pd.dummies) │    │ random=42    │    │ random=42    │    │ Visualization│
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Step 1 — Encoding

All categorical variables (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`) are transformed using **one-hot encoding** via `pd.get_dummies()`. This converts each category into a binary column, making the data suitable for the Decision Tree algorithm.

### Step 2 — Train/Test Split

The encoded dataset is split into **70% training** and **30% testing** sets using `train_test_split()` with `random_state=42` for reproducibility.

### Step 3 — Model Training

A `DecisionTreeClassifier` is initialized with:
- `max_depth=3` — limits tree depth for interpretability and to prevent overfitting
- `random_state=42` — ensures reproducible results

The model is trained on the training set using the `.fit()` method.

### Step 4 — Evaluation & Visualization

- **Accuracy Score** — overall correctness of predictions
- **Classification Report** — precision, recall, and F1-score for each class
- **Feature Importances** — top 5 most influential features
- **Tree Visualization** — full graphical representation saved as `decision_tree_model.png`

---

## 🚀 How to Run

### Prerequisites

```bash
pip install pandas scikit-learn matplotlib
```

### Execute

```bash
python task3.py
```

### Output

- Console: Accuracy score, classification report, top feature importances
- File: `decision_tree_model.png` — visualization of the trained decision tree

---

## 🛠️ Tech Stack

| Tool            | Purpose                     |
|-----------------|-----------------------------|
| **Python 3**    | Programming language         |
| **pandas**      | Data loading & preprocessing |
| **scikit-learn** | ML model & evaluation       |
| **matplotlib**  | Decision tree visualization  |

---

## 📁 Project Structure

```
SC_DS_3/
├── bank.csv                  # UCI Bank Marketing dataset
├── task3.py                  # Main Python script
├── decision_tree_model.png   # Tree visualization (generated)
├── linkedin_post.txt         # LinkedIn announcement
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 👤 Author

**Jinendra Banthia**
*Data Science Intern at SkillCraft Technology*

---

*Built with ❤️ as part of the SkillCraft Technology Data Science Internship Program.*
