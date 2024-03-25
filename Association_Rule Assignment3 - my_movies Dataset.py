# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 20:17:56 2024

@author: Priyanka
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules

movies=pd.read_csv("C:\Data Set\my_movies.csv")

frequent_itemsets=apriori(movies,min_support=0.03,max_len=4,use_colnames=True)
#items or item set must have minimum support value 3 % sale i.e.0.03
#you will get support values for 1,2,3 and 4 max items
#let us sort these support values
frequent_itemsets.sort_values('support',ascending=False,inplace=True)
#If we will  frequent_itemsets,70 % sale is of Gladiator,60 % sale is of sixth sense
#60% sale is of Patriot
#20% % sale is of Green Mile,LOTR1&LOTR2
#The lowest selling movies are Green Mile,LOTR1&LOTR2

rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1) 
#This generate association rules of size 1198X9 columns comprizes of antescends,consequences
# let us generate books having  mini.3 % sale
rules1=rules[rules['confidence']>0.6]
rules1
#The books which are sold 3 % minimum in single or in combination but with confidence=0.7,0.8,0.9 and 1

#The books with confidence 70% genereate 0 rules
rules2=rules1[rules1['confidence']==0.8]
rules2
