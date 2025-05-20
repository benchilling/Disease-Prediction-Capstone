Stroke Prediction Analysis
Healthcare Analytics Final Report
1. Problem Statement
Stroke is a leading cause of death and disability worldwide, with early detection and prevention being critical for reducing its impact. This project aims to develop a predictive model that can identify individuals at high risk of stroke based on their demographic information, medical history, and lifestyle factors. By accurately predicting stroke risk, healthcare providers can implement targeted preventive interventions for high-risk patients, potentially saving lives and reducing healthcare costs.
The challenge lies in building a model that can effectively handle the significant class imbalance in the data (only about 5% of cases are positive for stroke) while maintaining both precision and recall. Additionally, the model needs to provide interpretable insights that can guide clinical decisions.
2. Model Outcomes or Predictions
Type of Learning: Classification (Supervised Learning)
Expected Output: Binary classification (1 = stroke, 0 = no stroke)
Our models predict the probability of a patient experiencing a stroke based on their health data. We've implemented three supervised learning algorithms: Random Forest, Support Vector Machine (SVM), and Logistic Regression. Each model provides probability estimates that can be thresholded to optimize for different clinical objectives (e.g., maximizing recall to identify most at-risk patients, or balancing precision and recall for resource-limited settings).
3. Data Acquisition
The analysis uses the Stroke Prediction Dataset, containing healthcare information for 5,110 patients with the following features:

Demographic features: gender, age
Medical history: hypertension, heart disease, average glucose level, BMI
Lifestyle factors: smoking status, work type, residence type, marital status
Target variable: stroke occurrence (binary: 0 = no stroke, 1 = stroke)

The dataset exhibits significant class imbalance with only 4.87% of patients having experienced a stroke. This imbalance presents a challenge for model development and evaluation.

4. Data Preprocessing/Preparation
a. Handling Missing Values and Inconsistencies:

Identified missing values in the BMI column (201 records with 'N/A' values, about 3.93% of the data)
Implemented two strategies for comparison:

Removal: Dropped rows with missing BMI values for simplicity
Imputation: Used median imputation as an alternative approach in the code


Converted categorical variables to dummy variables using one-hot encoding
Standardized numerical features using StandardScaler to ensure all features contribute equally to the model

b. Training/Test Split:

Split the data using a 80/20 ratio (80% training, 20% testing)
Implemented stratified sampling to maintain the same class distribution in both sets
Used random seed (42) for reproducibility

c. Additional Analysis and Encoding:

Converted categorical variables to dummy variables using one-hot encoding
Applied feature scaling using StandardScaler
Implemented SMOTE (Synthetic Minority Over-sampling Technique) as an optional step to address class imbalance
Used cross-validation to ensure model robustness
Optimized decision thresholds for imbalanced classification

5. Modeling
We selected and implemented three different classification algorithms to predict stroke risk:

A. Random Forest:

Ensemble method using multiple decision trees to reduce overfitting
Can handle non-linear relationships and interactions between features
Provides feature importance measures for model interpretability
Hyperparameters tuned: n_estimators, max_depth, min_samples_split, min_samples_leaf, class_weight

B. Support Vector Machine (SVM):

Effective in high-dimensional spaces
Can handle complex decision boundaries
Less prone to overfitting in text classification problems
Hyperparameters tuned: C, kernel, gamma, class_weight

C. Logistic Regression:

Simple, interpretable model
Provides odds ratios that quantify risk factors
Efficient training and prediction
Hyperparameters tuned: C, penalty, solver, class_weight

Each model was implemented with class weights to address the significant class imbalance, and we used cross-validation to ensure reliable performance estimates.

6. Model Evaluation
Evaluation Metrics:
Given the significant class imbalance (4.87% stroke cases), we prioritized:

F1 Score - Harmonic mean of precision and recall
Recall - Ability to identify positive cases (critical for stroke detection)
ROC AUC - Overall discriminative ability
Precision-Recall AUC - Performance on the minority class

Model Selection:

Random Forest achieved the highest F1 score (0.252), providing the best balance between precision and recall
SVM showed the highest recall (0.897), making it effective at identifying stroke cases at the cost of more false positives
Logistic Regression achieved perfect recall (1.000) but with very low precision (0.051), resulting in the lowest F1 score

Feature Importance:

Age was consistently the most important predictor across all models
Glucose level, hypertension, and heart disease were also significant predictors
The odds ratios from Logistic Regression provided valuable clinical interpretability (e.g., age with an OR of 7.69, indicating dramatically increased stroke risk with age)

Final Model Selection:
The model choice depends on the specific clinical objective:

For balanced performance: Random Forest
For detecting the maximum number of stroke cases: Logistic Regression or SVM
For clinical interpretation and communication: Logistic Regression

I recommend the Random Forest model for general use due to its superior F1 score and ability to balance precision and recall.
