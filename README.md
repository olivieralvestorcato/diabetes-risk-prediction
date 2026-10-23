# Diabetes Risk Prediction

Machine learning classification project built with Python and scikit-learn using the Pima Indians Diabetes dataset.

The objective is to predict diabetes risk from clinical variables while comparing model performance using metrics that matter in a screening context, especially **recall**, **precision**, and **F1-score**.

## Project overview

The dataset contains **768 observations**, 8 predictive variables and a binary target (`Outcome`).

The workflow covers:

- Exploratory data analysis (EDA)
- Detection and treatment of medically implausible zero values
- Median imputation
- Feature scaling
- K-Nearest Neighbors (KNN) classification
- 5-fold cross-validation
- Hyperparameter tuning
- Feature selection
- Final model evaluation

## Data preparation

Several clinical variables contain zero values that are not medically plausible:

| Feature | Zero values |
|---|---:|
| Glucose | 5 |
| BloodPressure | 35 |
| SkinThickness | 227 |
| Insulin | 374 |
| BMI | 11 |

These values are converted to missing values and imputed using the median.

To prevent **data leakage**, preprocessing parameters are learned only from the training data. Features are standardized because KNN is a distance-based algorithm.

## Exploratory analysis

The correlation analysis shows that **Glucose** has the strongest relationship with the target (`Outcome`), with a correlation of approximately **0.49**.

Other notable relationships include:

- BMI ↔ SkinThickness: **0.65**
- Glucose ↔ Insulin: **0.58**

The feature-selection experiments later confirm the importance of Glucose: removing it causes the largest decrease in cross-validation F1-score.

## Baseline model

The first KNN model uses **K = 5**.

| Metric | Baseline KNN |
|---|---:|
| Accuracy | 71.4% |
| Precision | 58.7% |
| Recall | 67.3% |
| F1-score | 62.7% |

Confusion matrix:

```text
[[73 26]
 [18 37]]
```

## Cross-validation and feature selection

K is evaluated using **5-fold cross-validation** with F1-score as the optimization metric.

A scikit-learn pipeline is then used to perform median imputation, standardization and KNN classification correctly inside each validation fold.

Removing `SkinThickness` slightly improves cross-validation F1 from approximately **0.634 to 0.639**.

A leave-one-feature-out experiment shows that removing `Glucose` produces the largest performance drop, confirming that it is the most influential feature for this KNN model.

After feature selection, K is optimized again. The strongest cross-validation result is obtained with **K = 15**, reaching an F1-score of approximately **0.661**.

## Final model

The final pipeline uses:

```text
Median Imputation → StandardScaler → KNN (K=15)
```

with `SkinThickness` removed.

| Metric | Baseline KNN | Final KNN |
|---|---:|---:|
| Accuracy | 71.4% | **74.7%** |
| Precision | 58.7% | **66.0%** |
| Recall | **67.3%** | 60.0% |
| F1-score | 62.7% | **62.9%** |

Final confusion matrix:

```text
[[82 17]
 [22 33]]
```

## Key takeaway

The optimized model improves accuracy and precision, but recall decreases from **67.3% to 60.0%**.

This is important in a medical screening context. A false negative means that a diabetic patient is classified as non-diabetic. Therefore, the model with the highest accuracy is not automatically the most appropriate model for the real-world objective.

The baseline KNN detects a larger proportion of diabetic patients, while the optimized model is more conservative when predicting the positive class.

## Tech stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Jupyter Notebook

## Next steps

- Compare KNN with Logistic Regression and tree-based classifiers
- Add ROC-AUC and Precision-Recall analysis
- Optimize the classification threshold with a focus on recall
- Evaluate class-imbalance strategies where relevant
