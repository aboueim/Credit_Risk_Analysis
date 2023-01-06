# Credit_Risk_Analysis

## Overview of the Analysis

In this project, I intend to use supervised machine learning techniques to conduct a credit risk analyis on a credit card dataset from LendingClub, a peer-to-peer lending services company. In doing so, I will utlize Python and Jupyter Notebook, to read and run multiple classification analysis and then compare the obtained results. Specifically, after preprocessing the dataset, I will use LogisticRegression classifier to develop a predictive model and attempt to imporove its performance (given the imbalanced nature of loan risk status) through different resampling technqiues including over-, under-, and combined resampling. Next, I will apply easy ensemble and balanced random forests classifiers on the same dataset hoping to achieve higher predictive performance in comparison to the logisticregression. In the next sections, I provide the analyses results and conclude will a brief interpretation.

## Results

Below I added a screenshot of the results for each classifier and a short description of their performance.

### NaiveOverSampling
- balanced accuracy scores: 0.657
- precision
  * High Risk: 0.01
  * Low Risk: 1.00
- recall scores
  * High Risk: 0.60
  * Low Risk: 0.71
<img src="https://github.com/aboueim/Credit_Risk_Analysis/blob/main/Images/NaivOverSampling.png" width="600" height="400" />

### SMOTEOverSampling
- balanced accuracy scores: 0.662
- precision
  * High Risk: 0.01
  * Low Risk: 1.00
- recall scores
  * High Risk: 0.69
  * Low Risk: 0.63
<img src="https://github.com/aboueim/Credit_Risk_Analysis/blob/main/Images/SMOTEOverSampling.png" width="600" height="400" />

### UnderSampling
- balanced accuracy scores: 0.545
- precision
  * High Risk: 0.01
  * Low Risk: 1.00
- recall scores
  * High Risk: 0.40
  * Low Risk: 0.69
<img src="https://github.com/aboueim/Credit_Risk_Analysis/blob/main/Images/UnderSampling.png" width="600" height="400" />

### CombinedSampling
- balanced accuracy scores: 0.645
- precision
  * High Risk: 0.01
  * Low Risk: 1.00
- recall scores
  * High Risk: 0.57
  * Low Risk: 0.72
<img src="https://github.com/aboueim/Credit_Risk_Analysis/blob/main/Images/CombinedSampling.png" width="600" height="400" />

### RandomForests
- balanced accuracy scores: 0.789
- precision
  * High Risk: 0.03
  * Low Risk: 1.00
- recall scores
  * High Risk: 0.87
  * Low Risk: 0.70
<img src="https://github.com/aboueim/Credit_Risk_Analysis/blob/main/Images/RandomForests.png" width="600" height="400" />

### EasyEnsemble
- balanced accuracy scores: 0.932
- precision
  * High Risk: 0.09
  * Low Risk: 1.00
- recall scores
  * High Risk: 0.94
  * Low Risk: 0.92
<img src="https://github.com/aboueim/Credit_Risk_Analysis/blob/main/Images/EasyEnsemble.png" width="600" height="400" />

## Summary
