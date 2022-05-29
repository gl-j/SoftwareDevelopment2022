# Loan Predictability

Data set taken from: https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset
The data set is already presented in the train-test form

The data is from the company "Dream Housing Finance" which provides home loans.
In order for a customer to apply for a loan, first the company validates the customer eligibility for loan. Company wants to automate the loan eligibility process based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers.

This is a classification problem where it must be predicted if a loan would be approved or not

Variable | Description
----------|--------------
Loan_ID | Unique Loan ID
Gender | Male/ Female
Married | Yes/No
Dependents | Number of dependents (0, 1, 2 or +3)
Education | Graduate/ Not Graduate
Self_Employed | Yes/No
ApplicantIncome | Applicant income in USD
CoapplicantIncome | Coapplicant income in USD
LoanAmount | Loan amount * 1000 USD (if 1 then the loan amount is 1*1000 = 1000USD)
Loan_Amount_Term | Term of loan in days
Credit_History | credit history score (between 0.0 and 1.0)
Property_Area | Urban/ Semi Urban/ Rural
Loan_Status | Loan approved (Yes/No)
