#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:14:27 2019

@author: jai
"""

# Import
import numpy as np
import pandas as pd
import seaborn as sns 
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

# Data Preprocessing
data = pd.read_csv('kag_risk_factors_cervical_cancer.csv')

data.describe()

data.dtypes

data = data.replace('?', np.NaN)

data[['Number of sexual partners', 'First sexual intercourse',
      'Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
      'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD',
      'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis',
      'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
      'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
      'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
      'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
      'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Time since first diagnosis',
      'STDs: Time since last diagnosis']] = data[['Number of sexual partners', 
      'First sexual intercourse', 'Num of pregnancies', 'Smokes', 
      'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives', 
      'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)', 'STDs',
      'STDs (number)', 'STDs:condylomatosis', 'STDs:cervical condylomatosis', 
      'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis', 
      'STDs:syphilis', 'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
      'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
      'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Time since first diagnosis',
      'STDs: Time since last diagnosis']].apply(pd.to_numeric)



data.isnull().sum()

data['Number of sexual partners'] = data['Number of sexual partners'].fillna(round(data['Number of sexual partners'].mean()))
data['Smokes'] = data['Smokes'].fillna(round(data['Smokes'].mean()))
data['Smokes (years)'] = data['Smokes (years)'].fillna(round(data['Smokes (years)'].mean()))
data['Smokes (packs/year)'] = data['Smokes (packs/year)'].fillna(round(data['Smokes (packs/year)'].mean()))
data['First sexual intercourse'] = data['First sexual intercourse'].fillna(round(data['First sexual intercourse'].mean()))
data['Num of pregnancies'] = data['Num of pregnancies'].fillna(round(data['Num of pregnancies'].mean()))

data = data.drop(columns=['STDs: Time since first diagnosis',
                          'STDs: Time since last diagnosis'])

data = data[pd.notnull(data['STDs'])]

data['Hormonal Contraceptives'] = data['Hormonal Contraceptives'].fillna(round(data['Hormonal Contraceptives'].mean()))
data['Hormonal Contraceptives (years)'] = data['Hormonal Contraceptives (years)'].fillna(round(data['Hormonal Contraceptives (years)'].mean()))
data['IUD'] = data['IUD'].fillna(round(data['IUD'].mean()))
data['IUD (years)'] = data['IUD (years)'].fillna(round(data['IUD (years)'].mean()))

train_data = data.drop(['Hinselmann', 'Schiller', 'Citology', 'Biopsy'], axis=1)
test_data = data[['Hinselmann', 'Schiller', 'Citology', 'Biopsy']]

X_train, X_test, y_train, y_test = train_test_split(train_data, test_data, 
                                                    test_size=0.25, random_state=24)

# Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=300, random_state=0)  
classifier.fit(X_train, y_train)  
y_pred = classifier.predict(X_test)

y_test = y_test.reset_index()
y_test = y_test.drop(columns=['index'])

y_test = y_test.values

acc = 0
for i in range(1, 190):
    if(y_pred[i-1][0]==y_test[i-1][0]):
        acc += 1

print((acc/190)*100)


acc = 0
for i in range(1, 190):
    if(y_pred[i-1][1]==y_test[i-1][1]):
        acc += 1

print((acc/190)*100)


acc = 0
for i in range(1, 190):
    if(y_pred[i-1][2]==y_test[i-1][2]):
        acc += 1

print((acc/190)*100)


acc = 0
for i in range(1, 190):
    if(y_pred[i-1][3]==y_test[i-1][3]):
        acc += 1

print((acc/190)*100)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


confusion_matrix = confusion_matrix(y_test, y_pred)


single_predict = [['20','1','20','0','1','1','1','1','1','1','1','1','1','1','1', '1', 
                   '1','1','1','1','1','1','1','1','1','1','0','0','0','0','0','0', '0']]

single_predict_result = classifier.predict(single_predict)

# Multiple Classifier

multiple_target_classifier = MultiOutputClassifier(classifier, n_jobs=-1)

multiple_target_classifier.fit(X_train, y_train)  

y_pred = multiple_target_classifier.predict(X_test)

# EDA for Inusrance

cancer_data = data.loc[data['Dx:Cancer'] == 1]

sns.distplot(cancer_data['Age'])

sns.distplot(cancer_data['Number of sexual partners'])

sns.distplot(cancer_data['Smokes'])

sns.distplot(cancer_data['Num of pregnancies'])

sns.distplot(cancer_data['STDs'])

sns.distplot(cancer_data['First sexual intercourse'])

sns.distplot(cancer_data['Biopsy'])

# Save the Model
model_file_name = 'random_classifier_model_multiple.sav'
joblib.dump(multiple_target_classifier, model_file_name)

# Load the Model
loaded_model = joblib.load(model_file_name)
result = loaded_model.score(X_test, y_test)
print(result)
