# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 20:13:13 2024

@author: Priyanka
"""

"""
A retail store in India, has its transaction data, and it would 
like to know the buying pattern of the 
consumers in its locality, you have been assigned this task 
to provide the manager with rules 
on how the placement of products needs to be there 
in shelves so that it can improve the buying
patterns of consumes and increase customer footfall. 

"""

from mlxtend.frequent_patterns import apriori,association_rules
#Here we are going to use transactional data wherein size of each row is not consistent
#We can not use pandas to load this unstructured data
#here fuction called open() is used
#create an empty list
retail=[]

with open("C:/Data Set/transactions_retail1.csv") as f:transactions_retail1=f.read()

retail=transactions_retail1.split("\n")
#Earlier retail datastructure was in string format,now it will change to list of
#557042 ,each item is comma seperated
# our main aim is to calculate #A ,#C,we will have to seperate out each item from each transaction
retail_list=[]
for i in retail:
    retail_list.append(i.split(","))
# split fuction will seperate each item from each list,wherever it will find comma 
#it will split the item
# in order to generate association rules ,you can directly use retail_list
# Now let us seperate out each item from the retail list
all_retail_list=[ i for item in retail_list for i in item]    
# You will get all the items occured in all transactions
#We will get 3348059 items in various transactions


from collections import Counter
item_frequencies=Counter(all_retail_list)
#item_frequencies is basically dictionary having x[0] as key and x[1]=values

item_frequencies=sorted(item_frequencies.items(),key=lambda x:x[1])
item_frequencies

items=list(reversed([i[0] for i in item_frequencies])) 
#This is list comprehension for each item in item frequencies access the key

frequencies=list(reversed([i[1] for i in item_frequencies]))

import pandas as pd

#association rule mining

retail_series=pd.DataFrame(pd.Series(retail_list))

#we had extra row created,check the groceries_series ,last row is empty, let us first delete it
retail_series=retail_series.iloc[:557041,:]
#we have taken rows from 0 to 9834 and columns 0 to all
#groceries series has column having name 0,let us rename as transactions
retail_series.columns=["Transactions"]

x=retail_series['Transactions'].str.join(sep='*')
#check the x in variable explorer which has * seprator rather the ','
x=x.str.get_dummies(sep='*')

frequent_itemsets=apriori(x,min_support=0.0040,max_len=4,use_colnames=True)
#
#let us sort these support values
frequent_itemsets.sort_values('support',ascending=False,inplace=True)
#Support values will be sorted in descending order
#Even EDA was also have the same trend,in EDA there was count and here it is support value
#we will generate association rules,This association rule will calculate all the matrix
#of each and every combination
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1) 

rules=rules[["antecedents","consequents","support","confidence","lift"]]
rules.sort_values('lift',ascending=False).head(10)
rules1=rules[rules['support']<0.07]
rules1
rules2=rules1[rules1['confidence']==0.5]
rules2
