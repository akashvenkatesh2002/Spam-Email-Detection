# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 10:36:11 2022

@author: kavish
"""

#Import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#Import dataset
message=pd.read_csv('spam.csv',names=["labels","message"])

#Print first 4 datapoints to check if dataset is loaded properly 
print(message.head())

#General statistics
print(message.describe())
print(message.groupby('labels').describe())

#Creating a new column length for visualization of length of each datapoint
message['length']=message['message'].apply(len)
print(message.head())
message.length.describe()

#Histogram of frequency of each length
message['length'].plot(bins=50,kind='hist')


#Pre-processing : Building a function to remove all punctuations and stop words

#Demonstrating removal of punctuations
import string
mess = 'sample message!...'
nopunc=[char for char in mess if char not in string.punctuation]
nopunc=''.join(nopunc)
nopunc.split()
print(nopunc)

#Demonstrating examples of stopwords
from nltk.corpus import stopwords
print(stopwords.words('english')[0:10])


#Combining the above processes to create a filter function
def text_process(mess):
    nopunc =[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#Before and after
print(message.head())
print(message['message'].head(5).apply(text_process))

