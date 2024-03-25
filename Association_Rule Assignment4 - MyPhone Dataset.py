# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 19:53:41 2024

@author: Priyanka
"""


"""
A Mobile Phone manufacturing company wants to launch its 
three brand new phone into the market, 
but before going with its traditional marketing approach this time
it want to analyze the data of its previous model sales in different regions 
and you have been hired as an Data Scientist to help them out, 
use the Association rules concept and provide your insights to the company’s 
marketing team to improve its sales.
What is the business objective?
The main objective is to create a association rules to insights of earlier 
models to seller based on support,confidence and lift.
In addition to the Association rules Model prediction, we also have 
taken into account the  recommendation for a sale increase to mobile company.

Are there any constraints?
Understanding the metric for evaluation was a challenge as well.
Since the data consisted of binary data, EDA of  was a major challenge..


"""

import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules

mobile=pd.read_csv("C:/Data Set/myphonedata.csv")
mobile
#This loads data set
frequent_itemsets=apriori(mobile,min_support=0.03,max_len=4,use_colnames=True)

frequent_itemsets.sort_values('support',ascending=False,inplace=True)
#Support values will be sorted in descending order
#white color mobile was having higher sale with support value 0.63
#second rank mobile was red and blue with support value 0.54
#combi pack of white-red,red-blue,white-blue are having support value 0.36

rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1) 
rules=rules[["antecedents","consequents","support","confidence","lift"]]
rules.sort_values('support',ascending=False).head(10)
#This generate association rules of size 1198X9 columns comprizes of antescends,consequences
# let us generate mobile having  mini.3 % sale
#let us check rules on confidence
rules1=rules[rules['confidence']==1]
rules1
#if a person is purchasing orange mobile then there are 100 % chances that he may purchase white
#if a person is purchasing white and green mobile then there are 100 % chances that he may purchase red
#if a person is purchasing orange +red mobile then there are 100 % chances that he may purchase white
