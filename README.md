# Credit_Risk_Analysis

## Overview of the Analysis

In this project, I intend to use supervised machine learning techniques to conduct a credit risk analyis on a credit card dataset from LendingClub, a peer-to-peer lending services company. In doing so, I will utlize Python and Jupyter Notebook, to read and run multiple classification analysis and then compare the obtained results. Specifically, after preprocessing the dataset, I will use LogisticRegression classifier to develop a predictive model and attempt to imporove its performance (given the imbalanced nature of loan risk status) through different resampling technqiues including over-, under-, and combined resampling. Next, I will apply easy ensemble and balanced random forests classifiers on the same dataset hoping to achieve higher predictive performance in comparison to the logisticregression. In the next sections, I provide the analyses results and conclude will a brief interpretation.

## Results

Below I added a screenshot of the results for each classifier and a short description of their performance.

### NaiveOverSampling
- balanced accuracy scores: **0.657**
- precision
  * High Risk: **0.01**
  * Low Risk: **1.00**
- recall scores
  * High Risk: **0.60**
  * Low Risk: **0.71**
<img src="https://github.com/aboueim/Credit_Risk_Analysis/blob/main/Images/NaivOverSampling.png" width="600" height="400" />

### SMOTEOverSampling
- balanced accuracy scores: **0.662**
- precision
  * High Risk: **0.01**
  * Low Risk: **1.00**
- recall scores
  * High Risk: **0.69**
  * Low Risk: **0.63**
<img src="https://github.com/aboueim/Credit_Risk_Analysis/blob/main/Images/SMOTEOverSampling.png" width="600" height="400" />

### UnderSampling
- balanced accuracy scores: **0.545**
- precision
  * High Risk: **0.01**
  * Low Risk: **1.00**
- recall scores
  * High Risk: **0.40**
  * Low Risk: **0.69**
<img src="https://github.com/aboueim/Credit_Risk_Analysis/blob/main/Images/UnderSampling.png" width="600" height="400" />

### CombinedSampling
- balanced accuracy scores: **0.645**
- precision
  * High Risk: **0.01**
  * Low Risk: **1.00**
- recall scores
  * High Risk: **0.57**
  * Low Risk: **0.72**
<img src="https://github.com/aboueim/Credit_Risk_Analysis/blob/main/Images/CombinedSampling.png" width="600" height="400" />

### RandomForests
- balanced accuracy scores: **0.789**
- precision
  * High Risk: **0.03**
  * Low Risk: **1.00**
- recall scores
  * High Risk: **0.87**
  * Low Risk: **0.70**
<img src="https://github.com/aboueim/Credit_Risk_Analysis/blob/main/Images/RandomForests.png" width="600" height="400" />

### EasyEnsemble
- balanced accuracy scores: **0.932**
- precision
  * High Risk: **0.09**
  * Low Risk: **1.00**
- recall scores
  * High Risk: **0.94**
  * Low Risk: **0.92**
<img src="https://github.com/aboueim/Credit_Risk_Analysis/blob/main/Images/EasyEnsemble.png" width="600" height="400" />

## Summary

Overall, from the above results, it can be realized that the logistic regression classifier performs moderately, in terms of balanced accuracy, under all of the four resampling techniques used (from 0.545 to 0.662). Besides, this classifier lacks the ability to precisely detect high risk loaners (precision = 0.01 in all four resamplings types), while performing perfectly in precisely classifying low risk loaners (precision = 1.00 in all four resampling types). This result is not surprising, however, given the abundance of low-risk loaners. Nonetheless, it reveals that neither of the resampling techniques could help significantly improving the predictive model in terms of precision. In terms of sensitivity of the model, one the other hand, it can be seen that SMOTEOverSampling provides the highest recall rate for High Risk loaners (recall = 0.69), whereas CombinedResampling gives the highest recall rate for the Low Risk loaners (recall = 0.72). These results also allude to the week to moderate capability of the classifier in predicting risky loaners, even though different resampling techniques hold slightly higher or weaker sensitivity. However, utlizing ensemble learning allows for further improvements in the model. Using the balanced random forest classifier increased the balanced accuracy to 0.789; 0.12 increase from the most accurate logit classifier. It also improved the model's precision from 0.01 to 0.03 (although not atremendous increase) and the sensitivity of the model against high-risk loaners to 0.87, a 0.18 increase from the highest sensitivity rate in logit classifiers. However, randomforest was not successful in improving the sensitivity of the model in prediciting low-risk loaners (recall = 0.7). The second ensemble learning classifier (i.e., the easy ensemble) improved the model performance to a much higher level. The balanced accuracy raised to 0.932, the precision for high-risk loaners increased to 0.09, and both recall rates improve as well (high risk recall = 0.94 and low risk recall = 0.92).
From the obtained results, it can be concluded that among the tested classifiers, the easy ensemble had the highest performance and could almost reliably distinguish high-risk loaners from low-risk ones. Given in our scenario, cost of giving loans to high risk customers is much higher than mistakenly disapproving low-risk customers, it is crucial that the classifier shows a high rate of sensitivity. That is, the model does not erroneously identify high risk customers as low-risk (false negative). Therefore, I recommend that if any of the classifiers are supposed to be used by the company, the best choice will be the easy ensemble as it affords a high level of sensitivity toward high risk customers (recall = 0.94).
