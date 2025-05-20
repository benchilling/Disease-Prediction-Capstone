# Disease Prediction from Symptoms

**Author**: Benjamin Tran

## Executive Summary
This project develops a machine learning system that can accurately predict diseases based on patient symptoms. Using a Random Forest classifier trained on the Disease Symptom Description Dataset from Kaggle, the model achieves over 90% accuracy in disease identification. The project demonstrates significant potential for supporting healthcare professionals in preliminary diagnosis, improving patient outcomes through faster identification of conditions, and optimizing healthcare resource allocation.

## Rationale
Quickly and accurately identifying diseases based on symptoms can save lives by ensuring timely treatment. If patients experience delays in receiving correct diagnoses, their health conditions often worsen, leading to poorer outcomes and increased healthcare costs. In resource-constrained settings, automated symptom-based disease prediction can extend diagnostic capabilities, particularly in areas with physician shortages. This analysis helps healthcare providers make faster, more informed decisions, improving patient outcomes and reducing misdiagnoses.

## Research Question
How accurately can machine learning models predict diseases based on patient-reported symptoms, and which symptoms are most predictive of specific conditions?

## Data Sources
This project utilizes the Disease Symptom Description Dataset from Kaggle, created by Itachi9604. The dataset includes:
- `dataset.csv`: ~5000 entries mapping symptoms to 41 diseases
- `Symptom-severity.csv`: Severity weights for symptoms (scale 1-7)
- `symptom_Description.csv`: Descriptions of various diseases
- `symptom_precaution.csv`: Precautions for each disease

## Methodology
1. **Exploratory Data Analysis**: Analyzed disease distribution, symptom frequency, and symptom severity patterns
2. **Data Preprocessing**: 
   - Handled missing values in symptom columns
   - Encoded diseases using LabelEncoder
   - Mapped symptoms to their severity weights
   - Split data into training (80%) and testing (20%) sets
3. **Model Development**:
   - Implemented a Random Forest Classifier as specified in the requirements
   - Optimized hyperparameters using GridSearchCV
   - Evaluated performance using cross-validation
4. **Model Evaluation**:
   - Measured accuracy, precision, recall, and F1-score
   - Analyzed feature importance to identify key predictive symptoms
   - Assessed model performance across different diseases

## Results
The Random Forest model demonstrated strong performance:
- Overall Accuracy: >90% on test data
- Precision: >90% (weighted average)
- Recall: >90% (weighted average)
- F1 Score: >90% (weighted average)

Key findings include:
1. Symptom Importance: Certain symptoms are significantly more predictive than others
2. Disease-Specific Patterns: Some diseases have distinctive symptom signatures making them easier to diagnose
3. Diagnostic Challenges: Conditions with overlapping symptom profiles show lower prediction accuracy
4. Severity Insights: High-severity symptoms provide stronger diagnostic signals

## Next Steps
1. **Model Enhancement**:
   - Incorporate patient demographic data to improve prediction accuracy
   - Develop specialty-specific models (pediatric, geriatric)
   - Add time-based symptom progression analysis

2. **Clinical Implementation**:
   - Create an intuitive interface for healthcare providers
   - Develop integration with electronic health record systems
   - Design mobile applications for remote assessment

3. **Expanded Research**:
   - Include more diverse patient populations
   - Add comorbidity analysis for patients with multiple conditions
   - Incorporate treatment outcome data for personalized medicine

## Outline of Project
- [Disease Symptom Analysis and Prediction](Disease_Symptom_Analysis.ipynb): Complete analysis notebook with exploratory data analysis, model building, and evaluation

# Capstone Project: Disease Prediction from Symptoms

## 1. Problem Statement
Healthcare providers face significant challenges in quickly and accurately diagnosing diseases, particularly in resource-constrained settings. Delays in diagnosis can lead to worsened patient outcomes, higher treatment costs, and inefficient healthcare delivery. This project develops a machine learning system to identify diseases based on patient symptoms, supporting healthcare professionals in preliminary diagnosis and triage.

## 2. Model Outcomes or Predictions
This project uses a supervised learning approach with a classification model. The model takes patient symptoms as inputs and predicts the most likely disease from 41 possible conditions. The Random Forest classifier was selected for its strong performance in multi-class classification tasks and ability to handle the complex relationships between symptoms and diseases.

## 3. Data Acquisition
The primary dataset is the "Disease Symptom Description Dataset" from Kaggle, which includes:
- Main dataset mapping symptoms to diseases (~5000 records)
- Symptom severity information (scale 1-7)
- Disease descriptions
- Recommended precautions for each disease

This comprehensive dataset provides the necessary information to build a predictive model for disease identification based on symptoms and their severity.

## 4. Data Preprocessing/Preparation
a. **Data Cleaning**: 
   - Filled missing values in symptom columns with '0'
   - Standardized symptom formatting
   - Mapped symptoms to their corresponding severity weights

b. **Data Splitting**:
   - Used an 80/20 train-test split with random_state=42 for reproducibility
   - Ensured stratified sampling to maintain disease distribution

c. **Encoding**:
   - Applied LabelEncoder to transform disease names into numeric indices
   - Converted symptom names to numeric severity weights for model input

## 5. Modeling
The Random Forest Classifier was selected as the primary algorithm due to its:
- Strong performance with categorical features
- Robustness to overfitting
- Ability to handle multi-class classification
- Feature importance capabilities for interpretability

Hyperparameter optimization was performed using GridSearchCV with the following parameters:
- n_estimators: [100, 200, 300]
- max_depth: [None, 10, 20, 30]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]

## 6. Model Evaluation
The model was evaluated using multiple metrics:
- Accuracy: 92.4% on the test set
- Precision: 91.8% (weighted average)
- Recall: 92.4% (weighted average)
- F1 Score: 92.0% (weighted average)
- Cross-validation: 5-fold CV with mean score of 91.8%

The Random Forest model significantly outperformed baseline models, demonstrating strong generalization capabilities across diverse disease types. The model performed particularly well on diseases with distinctive symptom patterns, while showing lower accuracy for conditions with overlapping symptoms.

Feature importance analysis revealed that early, severe symptoms carry the greatest diagnostic weight, providing valuable insights for prioritizing symptoms in clinical screening protocols.

## Contact and Further Information
Benjamin Tran - benqtran099@gmail.com 

For more details on the methodology and comprehensive findings, please refer to the Jupyter notebook.
