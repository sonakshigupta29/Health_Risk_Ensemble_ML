# NovaGen - Health Risk Prediction using Ensemble Models

## 📌 Project Overview
Build an end-to-end **Supervised Machine Learning Pipeline** for predicting *health risk* using demographic, lifestyle, and clinical features. The project implements **Logistic Regression** as a *baseline model* alongside **Random Forest** and **Gradient Boosting** *Ensemble Classifiers*. **Binary classification** is performed with comprehensive **EDA, Data Preprocessing,** and **Model Evaluation** using **Recall, Accuracy,** and **ROC-AUC**. Special emphasis is placed on **recall** due to the health-critical nature of the problem, ensuring high-risk individuals are not missed.

## 🩺 Problem Statement
Early identification of unhealthy or high-risk individuals is crucial in healthcare applications.  
Missing a high-risk patient can lead to severe consequences; therefore, the model prioritizes **Recall** while maintaining strong overall performance.


## 📊 Dataset Description
The dataset contains health-related attributes including:

- Demographic features: Age  
- Clinical indicators: BMI, Blood Pressure, Cholesterol, Glucose Level, Heart Rate  
- Lifestyle factors: Sleep Hours, Exercise Hours, Water Intake, Smoking, Alcohol  
- Medical information: Stress Level, Mental Health, Medical History, Allergies  
- **Target Variable** (Health Outcome):  
  - `0` → Low Risk 
  - `1` → High Risk  

The dataset does **not contain missing values**, so only minimal preprocessing was required.


## 🔍 Exploratory Data Analysis (EDA)
EDA was performed to understand the dataset and uncover patterns:
- Distribution analysis of numerical features
- Count plots for categorical variables
- Feature-wise comparison with target variable(Health Outcome)
- Class distribution analysis & Outliers Detection
- Correlation heatmap for numerical attributes
- Multicollinearity and Skewness Check

*EDA insights helped guide model selection and evaluation strategy.*


## ⚙️ Data Preprocessing
- Encoding of categorical variables
- Feature scaling using `StandardScaler` (for Logistic Regression)
- Stratified Train-Test Split to preserve class distribution
- No heavy preprocessing or feature engineering was applied, as the focus of the project was on ensemble model comparison


## 🤖 Models Implemented
The following models were trained and evaluated:

1. **Logistic Regression (Baseline)**
   - Used as a linear baseline model
   - Implemented using a pipeline with feature scaling

2. **Random Forest Classifier**
   - Ensemble-based model
   - Provided the best overall performance without extensive tuning

3. **Tuned Random Forest Classifier**
   - Pre-pruning applied using hyperparameters such as `max_depth`, `min_samples_split`, and `min_samples_leaf` by GridSearchCV.

4. **Gradient Boosting Classifier**
   - Evaluated as an alternative ensemble approach.


## 📈 Evaluation Metrics
The models were evaluated using the following metrics:
- **Recall (Primary Metric)**  
- Accuracy  
- Precision  
- F1-score  
- Confusion Matrix  
- ROC-AUC (for the final selected model)
*Recall* was prioritized because minimizing false negatives is critical in health risk prediction.


## 🏆 Results Summary

| Model                          | Accuracy (%) |  Recall (%) |
|--------------------------------|--------------|-------------|
| Logistic Regression (Baseline) |   81.41      |   82.83     |
| Random Forest                  | **94.13**    | **96.48**   |
| Tuned Random Forest            |   93.45      |   95.68     |
| Gradient Boosting              |   93.03      |   94.97     |


## 📉 ROC-AUC Analysis
The final Random Forest model achieved a **ROC-AUC score of 0.984**, indicating excellent class separability and robust performance across different classification thresholds.


## ✅ Final Model Selection
**Random Forest** was selected as the final model due to:
- Highest recall score
- Strong accuracy and precision
- Excellent ROC-AUC score
- Robust generalization performance


## 🧠 Conclusion
This project demonstrates that ensemble-based models, particularly Random Forest, are highly effective for health risk prediction tasks. By prioritizing recall and validating performance using multiple evaluation metrics, the final model offers a reliable solution for identifying unhealthy individuals in a healthcare-oriented dataset.


## 🛠️ Technologies Used
- Python  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- Scikit-learn


## 🔮 Future Improvements
- Apply advanced feature engineering techniques
- Explore post-pruning using cost-complexity pruning
- Experiment with boosting algorithms such as XGBoost or LightGBM
- Perform model explainability using SHAP or permutation importance
- Investigate resampling techniques (e.g., SMOTE) for imbalance handling
- Extend the solution to a real-time or deployment-ready system


## ▶️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/health-risk-prediction.git

2. Navigate to the project directory:
   ```bash
   cd health-risk-prediction
   
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook

5. Open `NovaGen.ipynb` and run all cells sequentially.

**📌 Note**
This project was created primarily for learning and practicing ensemble machine learning models. Feature engineering was intentionally kept minimal to maintain focus on model comparison and evaluation.
