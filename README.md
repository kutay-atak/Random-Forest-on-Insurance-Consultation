\# Insurance Claim Modeling with Random Forest  

\### Frequency–Severity Approach



\## TL;DR

This project implements a \*\*two-part insurance risk model\*\*:



\- \*\*Frequency model\*\* → predicts whether / how often a claim occurs  

\- \*\*Severity model\*\* → predicts the size of a claim  



Both are built using \*\*Random Forest\*\*, reflecting how real-world insurance pricing separates risk into occurrence and impact.



\---



\## Problem



Insurance risk is not a single prediction problem.



Instead, expected loss is typically modeled as:



> \*\*Expected Cost = Frequency × Severity\*\*



Where:

\- Frequency = probability (or count) of claims  

\- Severity = cost per claim  



Most simple ML projects ignore this structure.  

This project explicitly models both components.



\---



\## Data



Two datasets are used:



\- `frequency.csv` → claim occurrence data  

\- `severity.csv` → claim size data  



\### Features

\- `uwYear` (underwriting year)

\- `gender`

\- `carType`

\- (additional structured risk features)



\### Targets

\- Frequency target → claim occurrence / count  

\- Severity target → claim size  



\---



\## Methodology



\### 1. Preprocessing

\- Categorical encoding using `OneHotEncoder`

\- Column-wise transformation via `ColumnTransformer`

\- Integrated into a `Pipeline` for reproducibility



\---



\### 2. Modeling



\#### Frequency Model

\- Model: `RandomForestClassifier`

\- Objective: predict claim occurrence / likelihood



\#### Severity Model

\- Model: `RandomForestRegressor`

\- Objective: predict claim size conditional on a claim



\---



\### 3. Why This Approach



This structure reflects real actuarial modeling:



\- Separates \*\*probability of risk\*\* from \*\*impact of risk\*\*

\- Handles skewed distributions more effectively

\- Allows more granular analysis of drivers



\---



\## Key Strengths



\- Uses \*\*pipeline-based architecture\*\* (clean + reproducible)

\- Applies \*\*ensemble methods\*\* for nonlinear relationships

\- Implements \*\*industry-relevant modeling structure\*\* (frequency–severity split)



\---



\## Limitations



\- No explicit hyperparameter tuning yet  

\- Models evaluated independently (not combined into expected loss)  

\- Limited interpretability without feature importance / SHAP  



\---



\## Next Steps (High Impact)



To make this a strong portfolio project:



\### 1. Combine models

Compute:

```python

expected\_cost = predicted\_frequency \* predicted\_severity

