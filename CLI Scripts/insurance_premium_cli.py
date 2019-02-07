#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 23:48:04 2019

@author: jai
"""

import pandas as pd

cancer_data = pd.read_csv('cancer_data')
cancer_data = cancer_data.loc[:, ~cancer_data.columns.str.contains('^Unnamed')]

print('##################################################')
print('\n')
print('Welcome!')
print('\n')
print('##################################################')

base_premium = int(input('Enter Base Insure Premium: '))

age = int(input('Enter age: '))

pregnancies = int(input('Enter 1 if pregnant, 0 otherwise: '))

stds = int(input('Enter 1 if any previous STD, 0 otherwise: '))

dx = int(input('Enter 1 if any previous Diagnosis, 0 otherwise: '))

if age >= 25 and age <= 45:
    age = '1'
else: 
    age = '0'
    
sum = int(age) + int(pregnancies) + int(stds) + int(dx)

premium = sum * 10

final_insurance_premium = int(base_premium + (base_premium * (sum / 10)))

print('##################################################')
print('\n')
print('The Final Insurance Premium is:', final_insurance_premium)
print('\n')
print('##################################################')
print('\n')
print('Thank You, this CLI is provided by Team Bits Please')
print('\n')
print('##################################################')
