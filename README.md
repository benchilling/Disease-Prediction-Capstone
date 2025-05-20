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

## Contact and Further Information
Benjamin Tran - benqtran099@gmail.com 

For more details on the methodology and comprehensive findings, please refer to the Jupyter notebook.
