#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 00:17:59 2019

@author: jai
"""

import pandas as pd
import seaborn as sns
from nltk import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('nightly-ClinicalEvidenceSummaries.tsv', delimiter='\t')

data = data[['disease', 'drugs', 'clinical_significance']]
data = data.dropna()

data['disease'].value_counts()
data['drugs'].value_counts()
data['clinical_significance'].value_counts()

# Frequency Counts

disease_freq = FreqDist()   
drug_freq = FreqDist()

for row in data.iterrows():
    disease_freq.update(row[1][0].replace(' ', '_').split())
    drug_freq.update(row[1][1].split(','))
    
print(drug_freq['key'])    

print(drug_freq['Cetuximab'])


drugs = pd.Series(drug_freq)
drugs = drugs.reset_index()
drugs.columns = ['drug', 'frequency']

diseases = pd.Series(disease_freq)
diseases = diseases.reset_index()
diseases.columns = ['disease', 'frequency']

highest_diseases = diseases.sort_values(by=['frequency'], ascending=False).iloc[:10]
highest_drugs = drugs.sort_values(by=['frequency'], ascending=False).iloc[:10]

# WordCloud

diseaseWords = ''

for row in data.iterrows():
    diseaseWords = diseaseWords + row[1][0] + ' '
    
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 10).generate(diseaseWords) 
                          
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 


drugsWords = ''

for row in data.iterrows():
    drugsWords = drugsWords + row[1][1] + ' '
    
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 10).generate(drugsWords) 
                          
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 


# Selecting most frequent values

truncated_data = data.loc[data['disease'].isin(highest_diseases['disease'])]
truncated_data = truncated_data.loc[truncated_data['drugs'].isin(highest_drugs['drug'])]

truncated_data = truncated_data[truncated_data.clinical_significance != 'Reduced Sensitivity']
truncated_data = truncated_data[truncated_data.clinical_significance != 'Adverse Response']

truncated_data = truncated_data.reset_index()
truncated_data = truncated_data.drop(columns=['index'])

one_hot_disease = pd.get_dummies(truncated_data['disease'])
truncated_data = truncated_data.drop('disease', axis = 1)
truncated_data = truncated_data.join(one_hot_disease)
 
one_hot_drugs = pd.get_dummies(truncated_data['drugs'])
truncated_data = truncated_data.drop('drugs', axis = 1)
truncated_data = truncated_data.join(one_hot_drugs)

one_hot_clinical_significance = pd.get_dummies(truncated_data['clinical_significance'])
truncated_data = truncated_data.drop('clinical_significance', axis = 1)
truncated_data = truncated_data.join(one_hot_clinical_significance)
 
truncated_data = truncated_data.drop(columns=['Resistance'])

truncated_data = truncated_data.reset_index()
truncated_data = truncated_data.drop(columns=['index'])

# Predict using Random Forest Classifier

X_train, X_test, y_train, y_test = train_test_split(truncated_data.iloc[:, :8], 
                                                    truncated_data['Sensitivity/Response'], 
                                                    test_size=0.25, random_state=24)

classifier = RandomForestClassifier(n_estimators=300, random_state=0)  
classifier.fit(X_train, y_train)  
y_pred = classifier.predict(X_test)

y_test = y_test.reset_index()
y_test = y_test.drop(columns=['index'])

acc = 0
for i in range(1, 18):
    if(y_pred[i-1]==y_test['Sensitivity/Response'][i-1]):
        acc += 1

print((acc/18)*100)


# Drugs with maximum success

successful_drugs_data = data.loc[data['clinical_significance'] == 'Sensitivity/Response']
successful_drugs_data = successful_drugs_data.drop(columns=['clinical_significance'])

successful_drugs_data = successful_drugs_data.reset_index()
successful_drugs_data = successful_drugs_data.drop(columns=['index'])

successful_drugs_data['drugs'].value_counts()

g = sns.countplot(successful_drugs_data['drugs'])
g.set(xticklabels=[])
   
successful_drug_freq = FreqDist()

for row in successful_drugs_data.iterrows():
    successful_drug_freq.update(row[1][1].split(','))

successful_drugs = pd.Series(successful_drug_freq)
successful_drugs = successful_drugs.reset_index()
successful_drugs.columns = ['drug', 'frequency']
    

highest_successful_drugs = successful_drugs.sort_values(
                            by=['frequency'], ascending=False).iloc[:10]

highest_successful_drugs = highest_successful_drugs.reset_index()
highest_successful_drugs = highest_successful_drugs.drop(columns=['index'])

g = sns.swarmplot(highest_successful_drugs['drug'], highest_successful_drugs['frequency'])
g.set_xticklabels(g.get_xticklabels(), rotation=30)
g.set_title('Safest & Best Drug Investment Options for Pharma Companies')


lowest_successful_drugs = successful_drugs.sort_values(
                            by=['frequency'], ascending=True).iloc[:10]

lowest_successful_drugs = lowest_successful_drugs.reset_index()
lowest_successful_drugs = lowest_successful_drugs.drop(columns=['index'])

print('High Risk & High Reward Drug Investment Options for Pharma Companies')
lowest_successful_drugs['drug'].value_counts()

# Demand of drugs

highest_diseases_data = data.loc[data['disease'].isin(highest_diseases['disease'])]
highest_diseases_data = highest_diseases_data.loc[highest_diseases_data['clinical_significance'] == 'Sensitivity/Response']
highest_diseases_data = highest_diseases_data.drop(columns=['clinical_significance'])

print('Cancer & Melanoma are the Most Common Diseases with Successful Drug Usage')
highest_diseases_data['disease'].value_counts()

highest_diseases_drug = FreqDist()

for row in highest_diseases_data.iterrows():
    highest_diseases_drug.update(row[1][1].split(','))

highest_demand_drugs = pd.Series(highest_diseases_drug)
highest_demand_drugs = highest_demand_drugs.reset_index()
highest_demand_drugs.columns = ['drug', 'frequency']

highest_demand_drugs = highest_demand_drugs.sort_values(by=['frequency'], ascending=False).iloc[:10]

g = sns.swarmplot(highest_demand_drugs['drug'], highest_demand_drugs['frequency'])
g.set_xticklabels(g.get_xticklabels(), rotation=30)
g.set_title('Highest Demand Drug Investment Options for Pharma Companies')
