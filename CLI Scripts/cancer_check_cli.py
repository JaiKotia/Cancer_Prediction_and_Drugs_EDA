#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 20:52:00 2019

@author: jai
"""

from sklearn.externals import joblib
import numpy

numpy.warnings.filterwarnings('ignore')

print('Please begin by entering details about the patient')

age = input('Please enter patients age: ')
no_of_sexual_encounters = input('Enter number of sexual partners: ')
first_sexual_intercourse = input('Enter age of first sexual intercourse: ')
no_of_pregancies = input('Enter number of pregnancies: ')
smokes = input('Enter 1 if patient smokes, 0 otherwise: ')

if smokes == '1':
    smokes_year = input('Enter number of smokes per year: ')
    smoke_packs_year = input('Enter number of smoke packs per year: ')
else:
    smokes_year = '0'
    smoke_packs_year = '0'
    
hormonal_contraceptives = input('Enter 1 if patient takes Hormonal Contraceptives, 0 otherwise: ')

if hormonal_contraceptives == '1':
    hormonal_contraceptives_year = input('Enter number of Hormonal Contraceptives per year: ')
else:
    hormonal_contraceptives_year = '0'
    
iud = input('Enter 1 if patient has taken IUDs, 0 otherwise: ')

if iud == '1':
    iud_year = input('Enter number of IUDs per year: ')
else:
    iud_year = '0'    
    
stds = input('Enter 1 if the patient has any STDs, 0 otherwise: ')
        
if stds == '1':
    pass
else:    
    stds_no = '0'
    stds_condylomatosis = '0'
    stds_cervical_condylomatosis = '0'
    stds_vaginal_condylomatosis = '0'
    stds_vulvo_perineal_condylomatosis = '0'
    stds_syphilis = '0'
    stds_pelvic_inflammatory_disease = '0'
    stds_genital_herpes = '0'
    stds_molluscum_contagiosum = '0'
    stds_AIDS = '0'
    stds_HIV = '0'
    stds_Hepatitis_B = '0'
    stds_HPV = '0'
    stds_diagnosis = '0'
    
dx = input('Enter 1 if any previous diagnosis, 0 otherwise: ')

if dx == '1':
    dx_cancer = input('Enter 1 if previously diagnosed with Cancer, 0 otherwise: ')
    dx_cin = input('Enter 1 if previously diagnosed with CIN, 0 otherwise: ')
    dx_hpv = input('Enter 1 if previously diagnosed with HPV, 0 otherwise: ')
else:
    dx_cancer = '0'
    dx_cin = '0'
    dx_hpv = '0'

X_input = [[age, no_of_sexual_encounters, first_sexual_intercourse, no_of_pregancies,
           smokes, smokes_year, smoke_packs_year, hormonal_contraceptives, 
           hormonal_contraceptives_year, iud, iud_year, stds, stds_no,
           stds_condylomatosis, stds_cervical_condylomatosis, stds_vaginal_condylomatosis,
           stds_vulvo_perineal_condylomatosis, stds_syphilis, stds_pelvic_inflammatory_disease,
           stds_genital_herpes, stds_molluscum_contagiosum, stds_AIDS, stds_HIV,
           stds_Hepatitis_B, stds_HPV, stds_diagnosis, dx, dx_cancer, dx_cin, dx_hpv]]
    
model_file_name = 'random_classifier_model_multiple.sav'
loaded_model = joblib.load(model_file_name)
result = loaded_model.predict(X_input)
print('##################################################')
print('Output')
print('##################################################')
print('Hinselmann Prediction:', result[0][0])
print('Schiller Prediction:', result[0][1])
print('Citology Prediction:', result[0][2])
print('Biopsy Prediction:', result[0][3])
print('##################################################')
print('Thank You, this CLI is provided by Team Bits Please')
print('##################################################')


