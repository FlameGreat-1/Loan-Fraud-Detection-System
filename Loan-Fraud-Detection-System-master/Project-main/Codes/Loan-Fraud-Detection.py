
# coding: utf-8

# # PeerLoanKart - Loan Repayment Prediction Project

# ***
# _**Importing the required libraries & packages**_


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import ydata_profiling as pf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')


# _**Changing The Default Working Directory Path & Reading the Dataset using Pandas Command and displaying the first five observations in the DataFrame**_


os.chdir(' C:\Users\USER\Downloads\Datascience-Project-main\Project-main\Codes')
df = pd.read_csv('loan_data.csv')
df.head()


# ## Exploratory Data Analysis(EDA)

# _**Getting the shape of the DataFrame**_

df.shape


# _**Checking for the duplicate values in the DataFrame**_


df.duplicated().sum()


# _**Checking for the null values in all the columns from the DataFrame**_



df.isna().sum()


# _**Getting the Data types and Non-null count of all the columns from the DataFrame using <span style = 'background : green'><span style = 'color : white'> .info() </span> </span> statement**_


df.info()


# _**Getting the summary of various descriptive statistics for all the numeric columns in the DataFrame and transposing it for the better view**_



df.describe().T


# _**Getting the correlation matrix for all the numeric columns in the DataFrame**_


df.corr()


# _**Automated Exploratory Data Analysis (EDA) with ydata_profiling(pandas_profiling) exporting it as <span style="color:red">.html</span> file and Displaying the result**_



EDA_Report = pf.ProfileReport(df)
EDA_Report.to_file("EDA_Report.html")
EDA_Report


# ## Data Visualisation

# _**Plotting the Bar Graph with count of customers borrowed loan for various purposes with their repayment count got from `purpose`, `not.fully.paid` column and identifying the customer who not paid their loan based on the purpose of loan from the DataFrame and saving the PNG File**_



plt.rcParams['figure.figsize'] = 15,7
sns.set_style('darkgrid')
plot = sns.countplot(x = df['purpose'],hue = df['not.fully.paid'],palette = 'Set1')
plt.xticks(rotation = 90)
for p in plot.patches:
    plot.annotate(p.get_height(),(p.get_x() + p.get_width() / 2.0,p.get_height()),
                 ha = 'center',va = 'center',xytext = (0,5),textcoords = 'offset points')
plt.title('Count of Customer based on Loan Purpose and Repayment')
plt.savefig('Count of Customer based on Loan Purpose and Repayment.png')
plt.legend(title = "Loan Repayment", labels = ['Paid', 'Not Paid'])
plt.show()


# _**Plotting the Histogram individually to show the distribution of data for both Negative and Postive Credit Policy based on `credit.policy` and `fico` (Fair Isaac Corporation) which fairly represent the Credit Score of the customer and saving the PNG file**_



fig , (ax1, ax2) = plt.subplots(sharex = False, sharey = False, nrows = 2, ncols = 1)
df[df['credit.policy'] == 1]['fico'].hist(alpha = 0.5, bins = 30, color = 'b', ax = ax1)
df[df['credit.policy'] == 0]['fico'].hist(alpha = 0.7,bins = 30, color = 'r', ax = ax2)
ax1.set_title('Distribution of Credit Policy [1] & FICO')
ax2.set_title('Distribution of Credit Policy [0] & FICO')
plt.subplots_adjust(hspace = 0.2)
plt.savefig('Data Distribution based on Credit Policy and FICO Score.png')
plt.show()


# _**Plotting the Histogram individually to show the distribution of data for Loan Repayment Status based on `not.fully.paid` and `fico` (Fair Isaac Corporation) which fairly represent the Credit Score of the Customer and saving the PNG file**_



fig , (ax1, ax2) = plt.subplots(sharex = False, sharey = False, nrows = 2, ncols = 1)
df[df['not.fully.paid'] == 1]['fico'].hist(alpha = 0.5, bins = 30, color = 'b', ax = ax1)
df[df['not.fully.paid'] == 0]['fico'].hist(alpha = 0.7,bins = 30, color = 'r', ax = ax2)
ax1.set_title('Distribution of Not Fully Paid & FICO')
ax2.set_title('Distribution of  Fully Paid & FICO')
plt.subplots_adjust(hspace = 0.2)
plt.savefig('Data Distribution based on Loan Repayment Status and FICO Score.png')
plt.show()


# _**Displaying the value counts of Customer's Loan Purpose based on `purpose` column from the DataFrame and Plotting the Pie-Chart based on the `purpose` column from the DataFrame to show the Percentage of Customer's Loan Purposes and saving the PNG file**_



print('Purpose of Loan taken by Customers ')
display(df['purpose'].value_counts())
print('*'*100)
data = df['purpose'].value_counts().values
labels = df['purpose'].value_counts().index
plt.pie(data, labels = labels, autopct = "%1.2f%%")
plt.title('Pie Representation on Percentage of Loan Purpose')
plt.savefig('Percentage of Loan Purpose.png')
plt.show()


# _**Displaying the value counts of Customer's Loan Repayment based on `not.fully.paid` column from the DataFrame and Plotting the Pie-Chart based on the `not.fully.paid` column from the DataFrame to show the Percentage of Customer's Loan Repayment Status and saving the PNG file**_



print("Customer's Loan Repayment")
Payment_History = df['not.fully.paid'].value_counts()
Payment_History.index = ['Paid','Not Paid']
display(Payment_History)
print('*'*100)
labels = 'PAID','NOT PAID'
sizes = [df['not.fully.paid'][df['not.fully.paid']==0].count(), df['not.fully.paid'][df['not.fully.paid']==1].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(8,5))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90)
ax1.axis('equal')
plt.title("Proportion of Customer with Paid and Not Paid", size = 20)
plt.savefig("Proportion of Customer with Paid and Not Paid.png")
plt.show()


# _**Getting the Correlation Values from all the numeric columns from the DataFrame using Seaborn Heatmap & saving the PNG File**_

sns.heatmap(df.corr(), cmap = 'coolwarm', square = True, annot = True, cbar = True)
plt.title('Correlation Heat Map')
plt.savefig('Correlation Heat Map.png')
plt.show()


# ## Data Cleaning

# _**One Hot Encoding the `purpose` column from the DataFrame using pandas get dummies function and displaying the DataFrame after One Hot Encoding**_


df1 = pd.get_dummies(df, columns = ['purpose'], drop_first = True)
df1.head()


# _**Assigning the dependent and independent variable**_



x = df1.drop(['not.fully.paid'], axis = 1)
y = df1['not.fully.paid']


# _**Defining the Function for the ML algorithms using <span style="color:purple">GridSearchCV</span> Algorithm and splitting the dependent variable & independent variable into training and test dataset and Predicting the Dependent Variable by fitting the given model and creating the pickle file of the model with the given Algo_name. Further getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset.**_


def Fitmodel(x, y, algo_name, algorithm, params, cv):
    np.random.seed(50)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,random_state = 95)
    grid = GridSearchCV(algorithm, params, scoring = 'accuracy', n_jobs = -1, cv = cv, verbose = 0)
    model = grid.fit(x_train, y_train)
    pred = model.predict(x_test)
    best_params = model.best_params_
    pickle.dump(model, open(algo_name,'wb'))
    cm = confusion_matrix(pred, y_test)
    print('Algorithm Name : ',algo_name,'\n')
    print('Best Params : ',best_params,'\n')
    print('Percentage of Accuracy Score : {0:.2f} %'.format(100*(accuracy_score(y_test,pred))),'\n')
    print('Classification Report : \n',classification_report(y_test,pred))
    print('Confusion Matrix : \n',cm,'\n')


# _**Running the function with empty parameters since the <span style = 'background : green'><span style = 'color : white'> Logistic Regression </span> </span> model doesn't need any special parameters and fitting the Logistic Regression Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name Logistic Regression.**_


params = {}
Fitmodel(x,y,'Logistic Regression',LogisticRegression(),params,cv = 10)


# _**Running the function with empty parameters since the <span style = 'background : green'><span style = 'color : white'> Gaussian Naive Bayes </span> </span> model doesn't need any special parameters and fitting the Gaussian Naive Bayes Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name GaussianNB**_


params = {}
Fitmodel(x,y,'GaussianNB',GaussianNB(),params,cv = 10)


# _**Running the function with empty parameters since the <span style = 'background : green'><span style = 'color : white'> Bernoulli Naive Bayes </span> </span> model doesn't need any special parameters and fitting the Bernoulli Naive Bayes Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name BernoulliNB**_


params = {}
Fitmodel(x,y,'BernoulliNB',BernoulliNB(),params,cv = 10)


# _**Running the function with empty parameters since the <span style = 'background : green'><span style = 'color : white'> Multinomial Naive Bayes </span> </span> model doesn't need any special parameters and fitting the Multinomial Naive Bayes Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name MultinomialNB**_



params = {}
Fitmodel(x,y,'MultinomialNB',MultinomialNB(),params,cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> KNeighbors Classifier </span> </span> Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name KNeighbors**_


params = {'n_neighbors' : [5,7,9,11,13,15],
          'p' : [1,2]}
Fitmodel(x,y,'KNeighbors',KNeighborsClassifier(),params, cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> SVC </span> </span> Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name SVC**_


params = {'C' : [0.01,0.1,1],
          'gamma' : [0.005,0.05,0.01]}
Fitmodel(x,y,'SVC',SVC(),params, cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> Decision Tree Classifier </span> </span> Algorithm and getting the Algorithm Name, Best Parameters of the algorithm,  Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name Decision Tree**_


params = {'criterion' : ['gini','entropy']}
Fitmodel(x,y,'Decision Tree',DecisionTreeClassifier(),params, cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> Random Forest Classifier </span> </span> Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name Random Forest**_


params = {'n_estimators' : [111,222,333,444,555],
          'criterion' : ['entropy','gini']}
Fitmodel(x,y,'Random Forest',RandomForestClassifier(), params, cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> Extra Trees Classifier </span> </span> Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name Extra Trees**_


params = {'n_estimators' : [111,222,333,444,555],
          'criterion' : ['entropy','gini']}
Fitmodel(x,y,'Extra Trees',ExtraTreesClassifier(),params, cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> Gradient Boosting Classifier </span> </span> Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name Gradient Boost.**_


params = {'n_estimators' : [111,222,333,444,555],
          'criterion' : ['friedman_mse', 'squared_error'],
          'loss' : ['deviance', 'exponential']}
Fitmodel(x,y,'Gradient Boost',GradientBoostingClassifier(),params, cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> XGB Classifier </span> </span> Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name XG Boost**_

params = {'n_estimators' : [111,222,333,444,555]}
Fitmodel(x,y,'XG Boost',XGBClassifier(),params,cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> CatBoost Classifier </span> </span> Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name CatBoost**_


params = {'verbose' : [0]}
Fitmodel(x,y,'CatBoost', CatBoostClassifier(),params, cv = 10)


# _**Running the function with empty parameters since the <span style = 'background : green'><span style = 'color : white'> LightGBM Classifier </span> </span> model doesn't need any special parameters and fitting the LightGBM Classifier\ Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name LightGBM**_


params = {}
Fitmodel(x,y,'LightGBM',LGBMClassifier(),params, cv = 10)


# _**Resampling the dependent variable so that the dependent variable values get balanced and assigning the new name for resampled variable. Displaying the Dependent variable count before and after oversampling**_


print('Before Oversampling')
display (df['not.fully.paid'].value_counts())
sm = SMOTE(random_state = 95)
x_res, y_res = sm.fit_resample (x, y)
print('-'*100)
print('After Oversampling')
display (y_res.value_counts())


# _**Running the function with empty parameters since the <span style = 'background : green'><span style = 'color : white'> Logistic Regression </span> </span> model doesn't need any special parameters and fitting the Logistic Regression Algorithm with resampled dependent and independent variable and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name Logistic Regression.**_


params = {}
Fitmodel(x_res,y_res,'Logistic Regression',LogisticRegression(),params,cv = 10)


# _**Running the function with empty parameters since the <span style = 'background : green'><span style = 'color : white'> Gaussian Naive Bayes </span> </span> model doesn't need any special parameters and fitting the Gaussian Naive Bayes Algorithm with resampled dependent and independent variable and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name GaussianNB**_


params = {}
Fitmodel(x_res,y_res,'GaussianNB',GaussianNB(),params,cv = 10)


# _**Running the function with empty parameters since the <span style = 'background : green'><span style = 'color : white'> Bernoulli Naive Bayes </span> </span> model doesn't need any special parameters and fitting the Bernoulli Naive Bayes Algorithm with resampled dependent and independent variable and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name BernoulliNB**_


params = {}
Fitmodel(x_res,y_res,'BernoulliNB',BernoulliNB(),params,cv = 10)


# _**Running the function with empty parameters since the <span style = 'background : green'><span style = 'color : white'> Multinomial Naive Bayes </span> </span> model doesn't need any special parameters and fitting the Multinomial Naive Bayes Algorithm  with resampled dependent and independent variable and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name MultinomialNB**_


params = {}
Fitmodel(x_res,y_res,'MultinomialNB',MultinomialNB(),params,cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> KNeighbors Classifier </span> </span> Algorithm with resampled dependent and independent variable and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name KNeighbors**_


params = {'n_neighbors' : [5,7,9,11,13,15],
          'p' : [1,2]}
Fitmodel(x_res,y_res,'KNeighbors',KNeighborsClassifier(),params, cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> SVC </span> </span> Algorithm with resampled dependent and independent variable and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name SVC**_


params = {'C' : [0.01,0.1,1],
          'gamma' : [0.005,0.05,0.01]}
Fitmodel(x_res,y_res,'SVC',SVC(),params, cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> Decision Tree Classifier </span> </span> Algorithm with resampled dependent and independent variable and getting the Algorithm Name, Best Parameters of the algorithm,  Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name Decision Tree**_


params = {'criterion' : ['gini','entropy']}
Fitmodel(x_res,y_res,'Decision Tree',DecisionTreeClassifier(),params, cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> Random Forest Classifier </span> </span> Algorithm with resampled dependent and independent variable and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name Random Forest**_


params = {'n_estimators' : [111,222,333,444,555],
          'criterion' : ['entropy','gini']}
Fitmodel(x_res,y_res,'Random Forest',RandomForestClassifier(), params, cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> Extra Trees Classifier </span> </span> Algorithm with resampled dependent and independent variable and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name Extra Trees**_


params = {'n_estimators' : [111,222,333,444,555],
          'criterion' : ['entropy','gini']}
Fitmodel(x_res,y_res,'Extra Trees',ExtraTreesClassifier(),params, cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> Gradient Boosting Classifier </span> </span> Algorithm with resampled dependent and independent variable and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name Gradient Boost.**_


params = {'n_estimators' : [111,222,333,444,555],
          'criterion' : ['friedman_mse', 'squared_error'],
          'loss' : ['deviance', 'exponential']}
Fitmodel(x_res,y_res,'Gradient Boost',GradientBoostingClassifier(),params, cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> XGB Classifier </span> </span> Algorithm with resampled dependent and independent variable and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name XG Boost**_


params = {'n_estimators' : [111,222,333,444,555]}
Fitmodel(x_res,y_res,'XG Boost',XGBClassifier(),params,cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> CatBoost Classifier </span> </span> Algorithm with resampled dependent and independent variable and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name CatBoost**_


params = {'verbose' : [0]}
Fitmodel(x_res,y_res,'CatBoost', CatBoostClassifier(),params, cv = 10)


# _**Running the function with empty parameters since the <span style = 'background : green'><span style = 'color : white'> LightGBM Classifier </span> </span> model doesn't need any special parameters and fitting the LightGBM Classifier Algorithm with resampled dependent and independent variable and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name LightGBM**_


params = {}
Fitmodel(x_res,y_res,'LightGBM',LGBMClassifier(),params, cv = 10)


# _**Splitting the original dependent & independent variable into training and test dataset.Fitting the Gradient Boosting Classifier model with the original train dependent and train independent variable and displaying the Accuracy of Gradient Boosting, Percentage of Accuracy Score,  Classification Report and Confusion Matrix between the predicted value and dependent test dataset**_


np.random.seed(50)
x_train,x_test, y_train,y_test = train_test_split (x,y,test_size = 0.3,random_state = 95)
grade = GradientBoostingClassifier (criterion = 'friedman_mse', loss = 'exponential', n_estimators = 111)
fit = grade.fit (x_train, y_train)
accuracy = fit.score(x_test,y_test)
predict = fit.predict(x_test)
cmatrix = confusion_matrix (y_test, predict)
print ('Accuracy of Gradient Boosting : ',(accuracy))
print ('Percentage of Accuracy Score : {0:.2f} %'.format(100*(accuracy_score(y_test,predict))))
print ('Classification Report:\n',classification_report(y_test,predict))
print ('Confusion Matrix :\n',cmatrix)


# _**Finding the feature importances of all the columns in the independent variable with respect to Gradient Boosting Classification Model above predicted for the dimensional reduction process**_


importances = grade.feature_importances_
indices=np.argsort(importances)[::-1]
print('Feature Ranking.')
for f in range (x.shape[1]):
    print('Feature %s(%f)' %(list(x)[f],importances[indices[f]]))


# _**Plotting the Bar Graph to represent the Feature Importances of the Independent variable column from the Gradient Boosting Classification model before oversampling and saving the PNG file**_


pd.Series(grade.feature_importances_,index=x.columns).sort_values(ascending=False).plot(kind='bar',figsize=(10,5))
plt.title('Feature Importance of Gradient Boosting Before Oversampling')
plt.savefig('Feature Importance of Gradient Boosting Before Oversampling.png')
plt.show()


# _**Splitting the resampled dependent & independent variable into training and test dataset.Fitting the XGB Classifier model with the resampled train dependent and train independent variable and displaying the Accuracy of XG Boost, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted value and dependent test dataset**_


np.random.seed(50)
x_train,x_test, y_train,y_test = train_test_split (x_res,y_res,test_size = 0.3,random_state = 95)
xgbc = XGBClassifier(n_estimators = 333)
fit = xgbc.fit (x_train, y_train)
accuracy = fit.score(x_test,y_test)
predict = fit.predict(x_test)
cmatrix = confusion_matrix (y_test, predict)
print ('Accuracy of XGBoost : ', (accuracy))
print ('Percentage of Accuracy Score : {0:.2f} %'.format(100*(accuracy_score(y_test,predict))))
print ('Classification Report:',classification_report(y_test,predict))
print ('Confusion Matrix :\n',cmatrix)


# _**Finding the feature importances of all the columns in the independent variable with respect to XG Boost Classification Model above predicted for the dimensional reduction process**_

importances = xgbc.feature_importances_
indices = np.argsort(importances)[::-1]
print ("Feature Ranking:")
for f in range (x.shape[1]):
    print ("Feature %s (%f)"  %(list (x)[f],importances[indices[f]]))


# _**Plotting the Bar Graph to represent the Feature Importances of the Independent variable column from the XG Boost Classification model after oversampling and saving the PNG file**_


feat_imp = pd.DataFrame({'Feature': list(x), 'Gini importance': importances[indices]})
plt.rcParams['figure.figsize']= (12,12)
ax= sns.barplot(x ='Gini importance', y = 'Feature', data = feat_imp  )
ax.set (xlabel = 'Gini Importances')
plt.title('Feature Importance of XGBoost After Oversampling')
plt.savefig('Feature Importance of XGBoost After Oversampling.png')
plt.show()


# _**With respect to Feature Importance of the independent variable reducing the dimensions of independent variable for reducing the complexity of model fitting**_


feat_imp.index = feat_imp.Feature
feat_to_keep = feat_imp.iloc[:11].index
display (type(feat_to_keep),feat_to_keep)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> XGB Classifier </span> </span> Algorithm with resampled independent and dependent variable after dimensional reduction and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name XG Boost_resample**_

params = {'n_estimators' : [111,222,333,444,555]}
Fitmodel(x_res[feat_to_keep],y_res,'XG Boost_resample',XGBClassifier(),params,cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> Random Forest Classifier </span> </span> Algorithm with resampled independent and dependent variable after dimensional reduction and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name Random Forest_resample**_


params = {'n_estimators' : [111,222,333,444,555],
          'criterion' : ['entropy','gini']}
Fitmodel(x_res[feat_to_keep],y_res,'Random Forest_resample',RandomForestClassifier(), params, cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> Extra Trees Classifier </span> </span> Algorithm with resampled independent and dependent variable after dimensional reduction and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name Extra Trees_resample**_


params = {'n_estimators' : [111,222,333,444,555],
          'criterion' : ['entropy','gini']}
Fitmodel(x_res[feat_to_keep],y_res,'Extra Trees_resample',ExtraTreesClassifier(),params, cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> Gradient Boosting Classifier </span> </span> Algorithm with resampled independent and dependent variable after dimensional reduction and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name Gradient Boost_resample.**_


params = {'n_estimators' : [111,222,333,444,555],
          'criterion' : ['friedman_mse', 'squared_error'],
          'loss' : ['deviance', 'exponential']}
Fitmodel(x_res[feat_to_keep],y_res,'Gradient Boost_resample',GradientBoostingClassifier(),params, cv = 10)


# _**Running the function with some appropriate parameters and fitting the <span style = 'background : green'><span style = 'color : white'> CatBoost Classifier </span> </span> Algorithm with resampled independent and dependent variable after dimensional reduction and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name CatBoost_resample**_

params = {'verbose' : [0]}
Fitmodel(x_res[feat_to_keep],y_res,'CatBoost_resample', CatBoostClassifier(),params, cv = 10)


# _**Running the function with empty parameters since the <span style = 'background : green'><span style = 'color : white'> LightGBM Classifier </span> </span> model doesn't need any special parameters and fitting the LightGBM Classifier Algorithm with resampled independent and dependent variable after dimensional reduction and getting the Algorithm Name, Best Parameters of the algorithm, Percentage of Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset, also creating the pickle file with the name LightGBM_resample**_

params = {}
Fitmodel(x_res[feat_to_keep],y_res,'LightGBM_resample',LGBMClassifier(),params, cv = 10)


# _**Loading the pickle file with the algorithm which gives highest accuracy score**_


model = pickle.load(open('XG Boost','rb'))


# _**Predicting the independent test variable using the loaded pickle file and displaying the Percentage of Accuracy Score between Test dependent variable and predicted value and also best parameters of the loaded pickle file.**_


pred1 = model.predict (x_test)
print ('Percentage of Accuracy Score for Best Fitted Model of Test Data: {0:.2f} %'.format(100*(accuracy_score(y_test,pred1))))
print('Params for Best Fitted Model : ',model.best_params_)


# _**Predicting the Loan Repayment Status using the loaded pickle file and displaying the Percentage of Accuracy Score between whole dependent variable and predicted value**_

fpred = model.predict(x)
print ('Percentage of Accuracy Score for Best Fitted Model of Whole Data: {0:.2f} %'.format(100*(accuracy_score(y,fpred))))


# _**Making the Predicted value as DataFrame with column name and Mapping the Predicted values to appropriate loan status term for better readability**_


fpred_df = pd.DataFrame(fpred, columns = ['Predicted Loan Repayment Status'])
fpred_df['Predicted Loan Repayment Status'] = fpred_df['Predicted Loan Repayment Status'].map({0 : 'Paid', 1 : 'Not Paid'})


# _**Renaming the Dependent variable column from the Original given DataFrame for further processing and Mapping the values to appropriate loan status term for better readability**_


df = df.rename(columns = {'not.fully.paid' : 'Loan Repayment Status'})
df['Loan Repayment Status'] = df['Loan Repayment Status'].map({0 : 'Paid', 1 : 'Not Paid'})


# _**Concating the Original DataFrame after processing and Predicted DataFrame as Final DataFrame and displaying the first five observations in the Final DataFrame**_


final_data = pd.concat([df, fpred_df], axis = 1)
final_data.head()


# _**Exporting the Final DataFrame with Actual Loan Repayment Status and Predicted Loan Repayment Status to a CSV(Comma Seperated value) File**_


final_data.to_csv('Loan Status Prediction.csv', index = False)

