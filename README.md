# Employee Attrition Analysis — IBM HR Dataset

This project explores the factors that lead employees to leave a company, 
using machine learning to predict attrition and understand its key drivers. 
The analysis is based on the IBM HR Analytics dataset and covers the full 
pipeline from data cleaning to model interpretation, with the goal of 
providing actionable insights for workforce retention strategies.

## Introduction
Employee attrition is one of the most costly and disruptive challenges an organization can face, as replacing a single employee can cost between half and twice their annual salary once recruitment, onboarding, lost productiity, and institutional knowledge are accounted for. Beyond financial costs, high attrition can damage team morale, it can break continuity on long term projects, and can signal deeper structural issues within a company.

Understanding why employees leave is therefore not just an analytical exercise but a strategic priority. Traditional HR analysis tends to focus on descriptive statistics and "after the fact" reporting, which identifies trends only once damage is already done. A predictive approach instead allows organisations to flag at-risk employees early 
and intervene before resignations occur.

This project applies machine learning to the IBM HR Analytics dataset 
to build a predictive model for attrition, with the dual aim of 
achieving strong predictive performance and producing interpretable 
insights into the drivers of attrition. The work was originally 
developed to support a relative's research project and later expanded 
as personal practice to strengthen skills in data analysis, model 
interpretability, and applied machine learning.

## Methodology

### Data Exploration
The dataset contains 1,470 employee records with 35 features covering 
demographics, compensation, tenure, satisfaction levels, and job 
characteristics. Initial exploration focused on understanding the 
target variable distribution, identifying skewed features, detecting 
redundancy among related variables, and examining relationships 
between candidate predictors and attrition. Several visualisation 
techniques were used including distribution plots, violin plots for 
commute distance, proportion plots for satisfaction-related variables, 
and a correlation heatmap. This stage revealed a moderate class 
imbalance (roughly 84% retention, 16% attrition) and strong redundancy 
between income-related variables.

### Data Preparation
Variables with no informative value were removed, including unique 
identifiers, constants, and redundant payroll rates (DailyRate, 
HourlyRate, MonthlyRate) which were found to be alternative encodings 
of the same compensation construct already captured by MonthlyIncome. 
The target variable was binary-encoded, and the dataset was split 
80/20 using stratified sampling to preserve the attrition class 
distribution across training and validation sets.

Class imbalance was deliberately not addressed through resampling. 
Techniques like SMOTE distort the data distribution and harm 
probability calibration, particularly problematic in attrition where 
predicted probabilities are used for ranking employees by risk. 
Instead, imbalance was handled at the modelling level through class 
weighting, which penalises minority-class misclassification without 
altering the data.

### Feature Engineering
One engineered feature was added: `IncomePerLevel` (MonthlyIncome 
divided by JobLevel). This captures relative pay, which is more 
psychologically meaningful than absolute pay, as two employees can earn 
the same salary but experience very different satisfaction depending 
on their level. The feature was created after the train/validation 
split to enforce best practice and prevent data leakage.

### Preprocessing Pipelines
Two separate preprocessing pipelines were built using scikit-learn's 
ColumnTransformer: one with feature scaling for linear models, and 
one without for tree-based models which are invariant to monotonic 
transformations. Median imputation was used for numeric features 
(robust to skew) and most-frequent imputation for categorical 
features. Categorical variables were one-hot encoded with the first 
category dropped to avoid multicollinearity in linear models.

### Modelling
Five models were evaluated using 5-fold stratified cross-validation: 
Logistic Regression as an interpretable linear baseline, Random Forest 
(both unconstrained and shallow variants), XGBoost, and LightGBM. 
Multiple metrics were tracked — ROC-AUC for ranking quality, recall 
for attrition detection (the primary objective given the cost 
asymmetry of missed resignations), precision, and F1-score.

### Model Calibration
After model selection, a calibration curve revealed systematic 
overconfidence in predicted probabilities, a known side effect of 
class weighting. Platt scaling (sigmoid calibration) was applied 
post-hoc to correct the probability scale while preserving the model's 
discriminative ability. This step is essential when probabilities are 
used for ranking or threshold setting, not just classification.

### Interpretability
Multiple complementary interpretability techniques were used: 
logistic regression coefficients and odds ratios for directional 
interpretation, ungrouped and grouped SHAP values for global feature 
importance, permutation importance for performance-based validation, 
and category-level SHAP plots to decompose categorical effects. Each 
technique addresses a different question (direction vs magnitude vs 
predictive reliance), and using them together gives a richer picture 
than any single method alone.

### Sanity Check
A final diagnostic experiment compared three variants — dropping the 
engineered feature, keeping all three income-related features, and 
dropping the original MonthlyIncome — to verify that feature 
engineering decisions did not artificially inflate performance.

## Results

Logistic Regression emerged as the strongest model, achieving a 
cross-validated ROC-AUC of 0.831 and the highest recall among all 
models at 0.742. Tree-based ensembles (Random Forest, XGBoost, 
LightGBM) all achieved comparable ROC-AUC values but consistently 
prioritised precision over recall, missing a larger share of actual 
attrition cases.

This result was counterintuitive at first. More complex models are 
typically expected to outperform linear baselines, but in this 
context complexity improved precision more readily than recall — and 
recall is the metric that matters most when missed resignations are 
more costly than false alarms.

The key drivers of attrition identified through SHAP and odds ratios 
were OverTime (strongest single predictor), JobRole, MonthlyIncome 
and IncomePerLevel, BusinessTravel frequency, MaritalStatus, and 
YearsAtCompany. Calibration analysis confirmed that post-hoc Platt 
scaling successfully corrected the model's systematic overconfidence, 
producing probability estimates that align well with observed 
attrition frequencies.

## Conclusion

This project reinforced several important lessons. First, model 
complexity does not automatically translate to better performance — 
the right model is the one that aligns with the cost structure of 
the problem, which in attrition means prioritising recall. Second, 
discrimination and calibration are distinct properties: a model can 
rank well while still producing unreliable probability estimates, and 
both need separate evaluation. Third, interpretability is not 
optional in applied settings — for an HR team to act on a model's 
predictions, they need to understand not just who is at risk but why.

More broadly, the project highlighted the value of methodological 
rigour over methodological complexity. Stratified sampling, careful 
handling of data leakage, separate preprocessing pipelines for 
different model families, and the use of multiple complementary 
interpretability techniques all contributed more to the quality of 
the final result than any individual modelling choice.


## Technologies Used
- **Language:** Python
- **Libraries:** pandas, scikit-learn, LightGBM, SHAP, seaborn, matplotlib
- **Environment:** Kaggle Notebook

## How to Run
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
2. Place the CSV file in the same folder as the notebook
3. Open `EmployeeAttritionAnalysis.ipynb` in Jupyter Notebook or Kaggle
4. Run all cells in order

## Dataset
IBM HR Analytics Attrition Dataset — 1,470 employee records with 35 
features including age, department, job role, overtime, and monthly income.

## Contributing
This is a personal project. Suggestions and feedback are welcome; 
feel free to open an issue.
