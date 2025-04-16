# 🏦 Banking Churn Prediction

This project focuses on analyzing and predicting **customer churn** for a bank using machine learning. The objective is to identify customers who are likely to leave the bank, enabling proactive retention strategies.

---

## 📌 Problem Statement

Customer churn is a critical metric in the banking industry. By predicting which customers are at risk of leaving, banks can take timely actions to improve customer satisfaction and reduce churn-related revenue loss.

---

## 📊 Dataset

The dataset contains information about bank customers, such as:

- Customer demographics (Age, Gender, Geography)
- Bank account details (Credit Score, Balance, Tenure, etc.)
- Product usage (Number of products, Active status)
- Target variable: `Exited` (1 = Churned, 0 = Stayed)

---

## 🔍 Project Workflow

1. **Data Preprocessing**
   - Encoding categorical variables using `StringIndexer` and `OneHotEncoder`
   - Combining features using `VectorAssembler`
   - Using `Pipeline` for streamlined transformation

2. **Exploratory Data Analysis (EDA)**
   - Churn distribution and imbalance detection
   - Correlation heatmap and feature selection
   - Age, tenure, and geography visualizations

3. **Model Building**
   - Logistic Regression (with class weights + tuning)
   - Random Forest
   - Gradient Boosted Trees (GBTClassifier)

4. **Model Evaluation**
   - Accuracy, Precision, Recall, F1 Score
   - ROC-AUC
   - Threshold tuning for imbalanced classification

5. **Hyperparameter Tuning**
   - Cross-validation with ParamGridBuilder
   - Best model selection using ROC-AUC

---

## 🛠️ Tech Stack

- PySpark (MLlib)
- Pandas & NumPy
- Scikit-learn (for evaluation)
- Matplotlib & Seaborn (for visualizations)
- Jupyter Notebook

---

## 📈 Model Performance Summary

### ✅ Logistic Regression

| Approach                                      | AUC    | Best Threshold | Recall (Churn=1) | F1 Score (Churn=1) | Accuracy |
|----------------------------------------------|--------|----------------|------------------|--------------------|----------|
| Baseline Logistic Regression                 | 0.7773 | 0.5            | 0.21             | 0.29               | 0.81     |
| + Threshold Tuning                           | 0.7773 | **0.3**        | **0.56**         | **0.50**           | 0.79     |
| + Hyperparameter Tuning                      | 0.7799 | 0.5            | 0.16             | 0.25               | 0.82     |
| + Hyperparameter Tuning + Threshold Tuning   | 0.7799 | **0.3**        | **0.55**         | **0.50**           | 0.79     |
| Weighted Logistic Regression                 | 0.7825 | 0.5            | ~0.25            | –                  | ~0.82    |



## 🧪 Logistic Regression: Evaluation Summary

To evaluate churn prediction with Logistic Regression, we explored multiple techniques:

### ✅ 1. Baseline Model
- Trained using default hyperparameters and a threshold of 0.5
- AUC: 0.7773
- Very low recall for churn class (only 21%)

### ✅ 2. Threshold Tuning
- Adjusted decision threshold to increase recall for churners
- Best threshold = 0.3
- Recall improved to 56% with only a small drop in accuracy

### ✅ 3. Hyperparameter Tuning
- Used cross-validation to find optimal values for:
  - `maxIter`, `regParam`, `elasticNetParam`
- Slight AUC improvement to 0.7799
- Still had low recall at default threshold

### ✅ 4. Tuning + Threshold
- Combined the tuned model with threshold adjustment
- Best results at threshold 0.3:
  - Recall = 55%
  - F1 score = 0.50
  - Accuracy = 79%

### ✅ 5. Weighted Logistic Regression
- Introduced class weights to give more importance to churners (minority class)
- AUC: 0.7825
- Performed well even at threshold 0.5
- Can be further improved with threshold tuning

---

📌 **Conclusion:**
Threshold tuning significantly improves recall in imbalanced datasets.  
The combination of hyperparameter tuning + threshold adjustment provides the best balance between precision, recall, and AUC.
---

### 🌲 GBTClassifier

| Approach                                    | AUC    | Best Threshold | Recall (Churn=1) | F1 Score (Churn=1) | Accuracy |
|--------------------------------------------|--------|----------------|------------------|--------------------|----------|
| Baseline GBTClassifier                      | 0.8592 | 0.5            | 0.50             | 0.59               | 0.87     |
| + Threshold Tuning                          | 0.8592 | **0.3**        | **0.62**         | 0.59               | 0.84     |
| + Hyperparameter Tuning                     | 0.8690 | 0.5            | 0.48             | 0.59               | 0.87     |
| + Hyperparameter Tuning + Threshold Tuning  | 0.8690 | **0.35**       | **0.61**         | **0.62**           | 0.86     |



## 🌲 GBTClassifier: Evaluation Summary

GBTClassifier (Gradient Boosted Trees) consistently delivered the best performance among all models evaluated.

### ✅ 1. Baseline Model
- AUC: 0.8592
- Recall: 0.50
- Accuracy: 0.87
- Strong performance even without tuning

### ✅ 2. Threshold Tuning
- Lowered threshold to improve recall
- Best performance at threshold = 0.3
  - Recall = 0.62
  - F1 Score = 0.59
  - Accuracy = 0.84

### ✅ 3. Hyperparameter Tuning
- Tuned parameters: `maxDepth`, `maxBins`, `stepSize`, `maxIter`
- AUC improved to **0.8690**
- Retained high precision with improved ranking quality

### ✅ 4. Tuning + Threshold Tuning
- Applied threshold tuning to the tuned GBT model
- Best result at threshold = 0.35:
  - Recall = 0.61
  - F1 Score = **0.62**
  - Accuracy = 0.86

---

📌 **Conclusion:**
The GBTClassifier with hyperparameter tuning and threshold adjustment provided the **best balance** between precision, recall, and AUC — making it the top-performing model for churn prediction in this project.
---

### 🌳 Random Forest

| Approach                                    | AUC    | Best Threshold | Recall (Churn=1) | F1 Score (Churn=1) | Accuracy |
|--------------------------------------------|--------|----------------|------------------|--------------------|----------|
| Baseline Random Forest                     | 0.8459 | 0.5            | 0.38             | 0.52               | 0.87     |
| + Threshold Tuning                          | 0.8459 | **0.3**        | **0.53**         | 0.59               | 0.86     |
| + Hyperparameter Tuning                    | 0.8605 | 0.5            | 0.47             | 0.58               | 0.87     |
| + Hyperparameter Tuning + Threshold Tuning | 0.8605 | **0.35**       | **0.58**         | **0.61**           | 0.86     |


## 🌳 Random Forest Classifier: Evaluation Summary

Random Forest is a versatile ensemble model known for its robustness and interpretability. It performed consistently well across all evaluation metrics.

### ✅ 1. Baseline Model
- AUC: 0.8459
- Recall: 0.38 (at threshold 0.5)
- Strong overall accuracy, but low recall for churners

### ✅ 2. Threshold Tuning
- Threshold adjusted to prioritize recall
- Best recall (0.53) at threshold = 0.3
- F1 score improved to 0.59, with accuracy at 0.86

### ✅ 3. Hyperparameter Tuning
- Tuned: `numTrees`, `maxDepth`, `maxBins`
- AUC improved to **0.8605**
- Slight gain in recall and F1-score

### ✅ 4. Tuning + Threshold Tuning
- Best-performing Random Forest variant
- Optimal threshold = **0.35** 
  - Recall = **0.58**
  - F1 Score = **0.61**
  - Accuracy = 0.86

---

📌 **Conclusion:**
Random Forest delivered consistent and interpretable results. With tuning and threshold adjustment, it reached strong recall and F1 scores — making it a practical and reliable model for churn prediction.
---

## 📁 Folder Structure

```bash
Banking_Churn/
├── data/                  # Raw & processed data
├── notebooks/             # Jupyter notebooks for EDA and modeling
├── models/                # Saved models
├── visuals/               # Graphs and plots
├── banking_churn.py       # Main script (if applicable)
└── README.md              # Project overview and documentation