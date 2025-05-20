# Disease Symptom Analysis and Prediction

## Project Overview
This repository contains the analysis and predictive modeling of the Disease Symptom Description Dataset from Kaggle. The project focuses on developing a machine learning model that can identify diseases based on patient symptoms with high accuracy, providing valuable insights for healthcare professionals and systems.

## Background
Quickly and accurately identifying diseases based on symptoms can save lives by ensuring timely treatment. Delays in receiving the correct diagnosis often lead to worsening health conditions and poorer outcomes. This analysis helps healthcare providers make faster, more informed decisions, improving patient outcomes and reducing misdiagnoses.

## Dataset
The dataset used in this project is the ["Disease Symptom Description Dataset"](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset/data) created by Itachi9604 on Kaggle. It consists of:

- **Dataset.csv**: Contains ~5000 entries mapping symptoms to 41 diseases
- **Symptom-severity.csv**: Provides severity weights for symptoms (scale 1-7)
- **Symptom_Description.csv**: Contains descriptions for various diseases
- **Symptom_precaution.csv**: Lists precautions for each disease

## Methodology
1. **Exploratory Data Analysis (EDA)**:
   - Analyzed disease distribution and frequency
   - Examined symptom occurrence and co-occurrence patterns
   - Studied symptom severity distributions

2. **Data Preprocessing**:
   - Handled missing values in symptom columns
   - Encoded categorical variables
   - Mapped symptoms to severity weights
   - Split data into training and testing sets (80/20)

3. **Model Development**:
   - Implemented a Random Forest Classifier
   - Optimized hyperparameters using GridSearchCV
   - Evaluated performance using 5-fold cross-validation

4. **Model Evaluation**:
   - Calculated accuracy, precision, recall, and F1-score
   - Generated and analyzed confusion matrix
   - Identified key predictive features (symptoms)

## Results
Our Random Forest model achieved excellent performance metrics:
- **Accuracy**: >90% on test data
- **Precision**: >90% (weighted average)
- **Recall**: >90% (weighted average)
- **F1 Score**: >90% (weighted average)

### Key Findings:
1. **Symptom Importance**: Certain symptoms are far more predictive than others for specific diseases
2. **Disease-Specific Patterns**: Some diseases have distinctive symptom patterns making them easier to diagnose algorithmically
3. **Symptom Co-occurrence**: Significant patterns in symptom co-occurrence provide diagnostic hints
4. **Severity Insights**: Diseases ranked by symptom severity can help prioritize urgent cases

### Business and Health Insights:
1. **Improved Diagnostic Efficiency**: Potential to reduce diagnostic time and cognitive load on healthcare providers
2. **Resource Optimization**: Can help optimize physician time allocation and prioritize high-risk patients
3. **Accessibility Enhancement**: Particularly beneficial for remote or underserved areas with limited healthcare access
4. **Educational Value**: Model insights can enhance medical education on symptom recognition

## Future Work
1. **Model Enhancement**:
   - Incorporate additional patient data (demographics, medical history)
   - Develop specialty-specific models
   - Include time-series analysis for symptom progression
   
2. **Interface Development**:
   - Create user-friendly interfaces for clinical settings
   - Develop API for integration with Electronic Health Record (EHR) systems
   - Build mobile applications for remote assessment

3. **Expanded Analysis**:
   - Include more diverse patient populations
   - Add comorbidity analysis for patients with multiple conditions
   - Explore ensemble methods with other algorithms

## Repository Structure
```
├── notebooks/
│   └── Disease_Symptom_Analysis.ipynb  # Main analysis notebook
├── data/
│   ├── dataset.csv                     # Main dataset with diseases and symptoms
│   ├── Symptom-severity.csv            # Symptom severity information
│   ├── symptom_Description.csv         # Disease descriptions
│   └── symptom_precaution.csv          # Precautions for diseases
├── models/
│   ├── disease_prediction_model.pkl    # Trained Random Forest model
│   ├── disease_label_encoder.pkl       # Label encoder for diseases
│   └── symptom_severity_dict.pkl       # Symptom severity dictionary
├── reports/
│   └── disease_insights_report.md      # Comprehensive business and health insights
├── README.md                           # Project overview (this file)
└── requirements.txt                    # Required dependencies
```

## Requirements
- Python 3.8+
- scikit-learn 1.0+
- pandas
- numpy
- matplotlib
- seaborn
- plotly

## Usage
1. Clone this repository
2. Install required dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook for detailed analysis
4. Explore the trained model and insights in the reports directory

## Conclusion
This project demonstrates the significant potential of machine learning approaches for symptom-based disease prediction. The Random Forest model developed shows promise for healthcare applications, potentially improving diagnostic accuracy, reducing time to treatment, and optimizing resource allocation. The insights generated provide valuable information for healthcare professionals and system designers.
