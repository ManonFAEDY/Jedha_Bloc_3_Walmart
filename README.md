# 🛒 Walmart Weekly Sales Prediction

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-v1.0+-orange.svg)](https://scikit-learn.org/stable/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-purple.svg)](https://github.com/votre_profil/votre_repo)

> **Machine Learning model to estimate weekly sales for Walmart stores, helping to understand feature influence and plan future campaigns.**

---

## 📋 Table of Contents
- [Context](#-context)
- [Project Objective](#-project-objective)
- [Data & Features](#-data--features)
- [Technologies](#-technologies)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Model Assessment](#-model-assessment)
- [Installation & Execution](#-installation--execution)
- [Author](#-author)

---

## 🎯 Context

**Walmart Inc.**, founded by Sam Walton in 1962, is a global retail giant. The ability to accurately predict future sales is essential for inventory optimization, workforce management, and marketing campaign planning.

**Mission:** Develop a linear regression model to estimate **weekly sales (`Weekly_Sales`)** based on economic indicators and store-specific data.

---

## 🚀 Project Objective

This project aims to provide Walmart's marketing department with a predictive and interpretable model, divided into three main steps:

### 1. **Data Preprocessing & EDA**
- Prepare data for Machine Learning.
- Handle missing values in the target variable (`Weekly_Sales`).
- Transform the `Date` column into usable features (`Year`, `Month`, etc.).
- Manage outliers using the **3-sigma rule** for numerical features.

### 2. **Baseline Model (Linear Regression)**
- Establish a simple linear regression model.
- Evaluate its performance using regression metrics (MAE/RMSE).
- Interpret the coefficients to identify key factors influencing sales.

### 3. **Regularized Regression**
- Train regression models with **regularization (Ridge/Lasso)**.
- Use `GridSearchCV` to find the optimal hyperparameter $\alpha$ to **reduce overfitting** and improve model generalization.

---

## 📊 Data & Features

### Source
Dataset sourced from a **Kaggle** competition (adapted for this project).

### Key Variables
| Type | Columns (Examples) | Role |
| :--- | :--- | :--- |
| **Target (Y)** | `Weekly_Sales` | Variable to predict (Weekly Sales). |
| **Numerical (X)** | `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`, `Date` | Economic and environmental indicators. |
| **Categorical (X)** | `Store`, `Holiday_Flag` | Store-specific and event-specific features. |

---

## 🛠️ Technologies

The project was developed in Python within a local environment and consolidated into a single notebook.

### Languages & Libraries
```python
pandas              # Data cleaning and preparation (EDA)
numpy                # Numerical computing
matplotlib/seaborn   # Visualizations
sklearn (scikit-learn) # Machine Learning, Preprocessing, Evaluation
````

### Scikit-learn Modules Used

  - `LinearRegression` (Baseline Model)
  - `Ridge`, `Lasso` (Regularized Models)
  - `StandardScaler`, `OneHotEncoder` (Preprocessing)
  - `GridSearchCV` (Hyperparameter Optimization)
  - `mean_absolute_error`, `mean_squared_error` (Evaluation)

-----

## 📁 Project Structure

```
walmart-sales-prediction/
│
├── 📓 walmart_sales_analysis.ipynb       # Main notebook containing all the code
├── 📊 walmart_sales_data.csv             # Input dataset
├── 📝 README.md                          # This file
└── 🖼️ img/                              # Folder for exported charts and graphs
```

-----

## 🔬 Methodology

### Part 1: Data Preparation

1.  **Target Cleaning:** Dropping rows with missing values in `Weekly_Sales`.
2.  **Feature Engineering:** Extracting `Year`, `Month`, `Day`, `DayOfWeek` from the `Date` column.
3.  **Outlier Handling:** Removing rows where numerical features (`Temperature`, `Fuel_Price`, etc.) fall outside the **[Mean ± 3 \* Standard Deviation]** range.
4.  **ML Pipeline:** Creating a `ColumnTransformer` to apply:
      * **Standardization** to numerical variables.
      * **One-Hot Encoding** to categorical variables (`Store`, `Holiday_Flag`).

### Part 2 & 3: Modeling

1.  **Baseline:** Training a `LinearRegression` model on the prepared data.
2.  **Evaluation:** Calculating **MAE** and **RMSE** on both the train and test sets to diagnose overfitting.
3.  **Regularization:** Training **Ridge** and **Lasso** models using `GridSearchCV` to find the optimal $\alpha$ that minimizes test error and reduces the gap between Train/Test errors.

-----

## 📈 Model Assessment

### Metrics

  - **MAE (Mean Absolute Error):** Provides the average absolute error, easy to interpret in the monetary unit of sales.
  - **RMSE (Root Mean Square Error):** Penalizes large prediction errors more severely.

### Overfitting Strategy

Overfitting is estimated by the **$\text{MAE}_{\text{Test}} / \text{MAE}_{\text{Train}}$ ratio**. The goal is to achieve a ratio close to **1** while minimizing the overall $\text{MAE}_{\text{Test}}$. Regularization is the key technique used to achieve this balance.

-----

## 💻 Installation & Execution

To run this project, you only need a Python environment with the libraries listed in `Technologies`.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ManonFAEDY/walmart-sales-prediction.git](https://github.com/ManonFAEDY/walmart-sales-prediction.git)
    cd walmart-sales-prediction
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
3.  **Open the Notebook:**
    Launch the `walmart_sales_analysis.ipynb` notebook in your environment (Jupyter Lab/Notebook or VS Code).
    ```bash
    jupyter notebook walmart_sales_analysis.ipynb
    ```
4.  **Execute:** Run the cells sequentially to see the EDA, preprocessing, model training, and evaluation results.

-----

## 👤 Author

**[Manon FAEDY]**

  - 🐙 [GitHub](https://github.com/ManonFAEDY)

-----

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

-----

## 🙏 Acknowledgments

  - **Walmart Inc.** for the sales data.
  - **Jedha** for the project framework and structure.
  - **Kaggle** for the original dataset.

-----

\<div align="center"\>
  \<strong\>⭐ If this analysis was helpful, don't forget to star the repository\! ⭐\</strong\>
\</div\>

```
```
