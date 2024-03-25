# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 23:26:27 2024

@author: Priyanka
"""


"""
The Departmental Store, has gathered the data of the products it sells on a Daily basis.
Using Association Rules concepts, provide the insights on the rules and the plots.


"""

from mlxtend.frequent_patterns import apriori,association_rules
#create an empty list
groceries=[]
with open("C:/Data Set/groceries.csv") as f:transactions_retail1=f.read()

groceries=transactions_retail1.split("\n")

groceries_list=[]
for i in groceries:
    groceries_list.append(i.split(","))

all_groceries_list=[ i for item in groceries_list for i in item]    
# You will get all the items occured in all transactions
#We will get 3348059 items in various transactions

#Now let us count the frequency of each item
#we will import collections package which has Counter function which will count the items
from collections import Counter
item_frequencies=Counter(all_groceries_list)
#item_frequencies is basically dictionary having x[0] as key and x[1]=values

#Now let us sort these frquencies in ascending order
item_frequencies=sorted(item_frequencies.items(),key=lambda x:x[1])
item_frequencies

items=list(reversed([i[0] for i in item_frequencies])) 

frequencies=list(reversed([i[1] for i in item_frequencies]))


import pandas as pd

groceries_series=pd.DataFrame(pd.Series(groceries_list))

groceries_series=groceries_series.iloc[:9835,:]
#we have taken rows from 0 to 9834 and columns 0 to all
#groceries series has column having name 0,let us rename as transactions
groceries_series.columns=["Transactions"]

x=groceries_series['Transactions'].str.join(sep='*')
#check the x in variable explorer which has * seprator rather the ','
x=x.str.get_dummies(sep='*')

frequent_itemsets=apriori(x,min_support=0.0040,max_len=4,use_colnames=True)

frequent_itemsets.sort_values('support',ascending=False,inplace=True)

rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1) 
#This generate association rules of size 4472X5 columns comprizes of antescends,consequences
rules=rules[["antecedents","consequents","support","confidence","lift"]]
rules.sort_values('lift',ascending=False).head(10)
rules1=rules[rules['support']<0.04]
rules1
#The groceries which are sold 3 % minimum in single or in combination but
# with confidence=0.7,0.8,0.9 and 1
rules2=rules1[rules1['confidence']>0.7]
rules2
#1 if customer is buying root vegetables, yogurt then there is 70 % chances it will buy Whole milk
#2 if cutomer is buying butter, curd then there are 70 % chances that cutomer will buy whole milk
#if cutomer is buying domastic eggs, curd then there are 70 % chances that cutomer will buy whole milk
