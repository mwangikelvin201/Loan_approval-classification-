# Loan_approval-classification-
![IMG](https://github.com/mwangikelvin201/Loan_approval-classification-/blob/3bd47a2ebbbc198b08085f4bc986bb54c1b22024/istockphoto-1308841055-1024x1024.jpg)
# 1.BUSINESS UNDERSTANDING
The financial field has to be handled with a lot of sensitivity especially when lending money to customers.The primary goal of this loan approval model project is to automate and improve the decision-making process for approving or rejecting loan applications. This will be done by building a machine learning model that predicts the likelihood of a loan applicant defaulting on a loan based on historical data.

## Predict Loan Default Risk:
The model aims to predict whether a loan applicant is likely to repay the loan or default. This enables financial institutions to make informed lending decisions.

## Streamline Loan Approval Process:
Automating the process helps to reduce human error, save time, and scale the approval process for large volumes of applications.

## Improve Customer Experience:
Faster, more accurate loan decisions result in a better experience for customers, improving satisfaction and trust in the institution.

# 2. PROBLEM STATEMENT
Loan approval decisions are traditionally made using manual processes, which are time-consuming and subject to human biases. Furthermore, lenders may struggle with accurately predicting which applicants will repay their loans. The problem is to design a model that uses historical data to identify patterns and predict whether a borrower is likely to default on a loan.

## Key challenges:
Data quality and availability:
Access to clean, structured, and relevant historical data, such as credit scores, loan histories, and applicant details.

## Fairness and bias:
The model must avoid discrimination against certain demographics (e.g., age, gender, race). Regulatory compliance: Ensuring the model aligns with financial regulations such as fair lending practices.
#  3. SCOPE OF THE PROJECT
The scope defines what will and will not be included in the project.

In-Scope:
Data collection and analysis of historical loan approval data.

Building and training the machine learning model.

Model evaluation and validation (e.g., assessing accuracy, precision, recall, etc.).

out of scope
The development of new loan products.

Post-loan servicing or collection processes.

Data collection from external or non-traditional data sources (unless the company has access to them).

# 4. STAKEHOLDERS
## Business Owners/Executives:
Will use the model to make informed decisions on lending practices, risk assessment, and financial strategy.

## Credit Risk Analysts:
These professionals evaluate the creditworthiness of applicants. The model will assist them in their analysis and help standardize risk assessments.

## Data Science and IT Teams:
Responsible for building and deploying the model.

## Customers:
Loan applicants who are affected by the approval or rejection decisions. A more accurate model means fairer and more consistent decisions.

## Regulators:
They will need to ensure that the model complies with financial regulations and ethical standards.

# 5. METHODOLOGY 
## Model Selection:
The baseline model employed was the Sequential Model from Keras, It was followed by the XGBoost Model and the Random Forest Classifier as well as the Logistic regressor.

## Training and testing:
75% of the data was used for training the model and 25% was used for testing and validation.

## Evaluation metrics:
Having worked with an imbalanced dataset, the metrics used were F1Score, recall and precision.

# 6. SUCCESS CRITERIA

The model's success will be measured based on its ability to improve the loan approval process. Success metrics could include:

## Model Accuracy:
How well the model predicts whether a loan applicant will default or repay.

## Precision and Recall:
For better handling of false positives (approving loans that shouldn't be approved) and false negatives (rejecting loans that should be approved).

## Business Impact:
Improvements in key business metrics such as reduced loan defaults, faster processing times, or better customer retention. Regulatory Compliance: Ensuring the model adheres to relevant financial and privacy regulations. Scalability: Ability to handle a large number of loan applications.

# 7.DATA UNDERSTANDING¶

Data is the foundation of any predictive model. For the loan approval model, the data typically includes:

## Applicant Demographics:
Age, gender, marital status, employment status, income, etc.

## Credit History:
Credit score, previous loan information, repayment history, etc.

## Loan Details:
Loan amount, loan type, term length, etc.

## Behavioral Data:
Transactions, savings, and spending habits that may indicate financial responsibility. Economic Factors: Macroeconomic data such as unemployment rates, inflation, or regional economic conditions (optional, but helpful in some cases).

# DATA SOURCE
Kaggle datasets

# DATA INSIGHTS

The data had different predictors, encoded in different formats(float, integer and object)

We had a binary target variable. It had two classes 0 and 1 which represented not approved and approved respectively. 

# DATA VISUALIZATION

Some of the visualizations have been incooperated to give the general distribution of the data and performance of the model.

### Distribution of the credit score  as well as the outlier detection for the same.

![img](https://github.com/mwangikelvin201/Loan_approval-classification-/blob/05cd7b73d892d086a4b0d029da57ef513d825452/numerical.png)

### confusion matrix for the model

![img](https://github.com/mwangikelvin201/Loan_approval-classification-/blob/86f77713327be1ca264590f7bfa595673dd7604f/matrix.png)


## Here's the lowdown on that confusion grid:
### True Negatives (TN): 7943
These are loans where customers didn't miss payments, predicted accurately as such.

### False Positives (FP): 478
These are loans where customers kept up with payments but were wrongly tagged as defaulters.

### False Negatives (FN): 372
Loans in this bunch slipped through and should've been flagged as defaulting but weren't.

### True Positives (TP): 2116
In this case, loans expected to default were indeed caught by the model.

# 8.CONCLUSIONS

From the models and insights above, we can come to the following observations:
All the models have a high accuracy(89) adn above. However our dataset being very imbalanced, we cannot assume our models have a good performance from the accuracy alone.

Conveniently the precision ,recall and Fbetascore which are the guiding metrics of a model from imbalanced data,are very good too.

The XGBClassifier and randomforest models have very good performances of the three metrics.

Even after applying SMOTE to deal with the class imbalance still there is not a very big change in our guiding metrics indicating that the class imbalance does not affect the performance of the model in a much significant way.

The confusion matrix also visualizes the general predictive ability of the model and it is also indicates quite a high accurate prediction of the data.

# 9. RECOMMENDATIONS
Based on the insights derived from the loan default prediction model, here are tailored recommendations for the financial institution looking to reduce loan defaults and improve overall risk management as well as improve customer relations:

### i. Proactive Default Prevention Strategies
Proactive Risk Mitigation : This actively involves managing loans at risk of default.

This will include leveraging the model’s predictive power to identify high-risk loans early in their lifecycle (especially loans with a high likelihood of default). For these loans, initiate proactive engagement such as:

Offering restructured repayment plans.

Providing financial counseling or personalized assistance to help borrowers manage their payments.

Introducing temporary deferments or interest rate reductions for at-risk customers.

This approach will help the firm to share and manage risks with the borrower

### ii. Approve loans based on risk profiles:
Use the model’s predictions to fine-tune the loan approval process by incorporating default risk scores into decision-making. This will allow the company to:

Offer loans to the low risk customers who are not likely to default.

Deny loans to high risk borrowers or offer them with strict restrictions.

### iii. Adjustment of interest rates and securities for high risk individuals
This involves offering loans at a higher interest rate or more securities to high risk customers. Thuis will discourage them from defaulting to avoid the loss of their posessions used as security.

### iv. Personalized Loan Products
This involves tailoring of different loan packages to accomodate different individuals of distinct risk levels.


