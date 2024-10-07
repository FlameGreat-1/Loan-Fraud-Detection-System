# PeerLoanKart-Loan-Fraud-Detection-Datascience-Project
This project uses machine learning to predict whether a loan applicant will repay their loan. The project uses a dataset of historical loan data from PeerLoanKart, a peer-to-peer lending platform. 
### Project Domain : Banking/Loan
## Problem Statement :
`Business Requirement` - PeerLoanKart is an NBFC (Non-Banking Financial Company) which facilitates peer to peer loan.
It connects people who need money (borrowers) with people who have money (investors). As an investor, you would want to invest in people who showed a profile of having a high probability of paying you back.

`Key issues` - Ensure NPAs are lower – meaning PeerLoanKart wants to be very diligent in giving loans to borrower
## Solution Approach :
The aim is to create a model that will help predict whether a borrower will pay the loan or not. The main focus is on Lower NPA (Non Performing Asset) to creating the model. 
## Observation : 
The following observations were made during the project period. 
1. The dataset of historical loan data from PeerLoanKart is a valuable resource for loan repayment prediction.
2. A variety of classification algorithms can be used to predict loan repayment status.
3. The model performance can be improved by using Feature Engineering and Hyperparameter tuning.
### Dataset Decription : 
  * credit.policy: 1 if the customer meets the credit underwriting criteria of PeerLoanKart, and 0 otherwise
  *	purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other")
  *	int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by PeerLoanKart to be more risky are assigned higher interest rates
  *	installment: The monthly installments owed by the borrower if the loan is funded
  *	log.annual.inc: The natural log of the self-reported annual income of the borrower
  *	dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income)
  *	fico: The FICO credit score of the borrower
  *	days.with.cr.line: The number of days the borrower has had a credit line
  *	revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle)
  *	revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available)
  *	inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months
  *	delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years
  *	pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments)
  *	not.fully.paid: This is the output field. Please note that 1 means borrower is not going to pay the loan completely
## Findings : 
The following findings were made during the project period.
1. Machine learning can be used to predict whether a loan applicant will repay their loan with a high degree of accuracy.
2. The most important features for predicting loan repayment are the applicant's credit score, credit policy, interest rates, installments, annual income and debt-to-income ratio.
3. The model can be used to help lenders make more informed decisions about whether to approve a loan application.
## Insights : 
The developed model can be used to improve the efficiency of loan lending by helping PeerLoanKart to make more informed decisions about whether to approve or deny loan applications. Here the model used is Extreme Gradient Boosting(XG Boost) model with the 94% degree of accuracy.

`Business Benefits` - Increase in profits up to 20% as NPA will be reduced due to loan disbursal for only good borrowers.
## Conclusion :
After predicting the loan repayment status, changing the categorical values to 'Paid' and 'Not Paid' based on the  `not.fully.paid` column and also name of the column to loan repayment status in original dataset and concating the predicted loan repayment status similar to loan repayment status for better identifications and readability in the Final Predicted Data
