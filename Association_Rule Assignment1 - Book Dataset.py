# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 23:24:39 2024

@author: Priyanka
"""


"""
Kitabi Duniya, a famous book store in India,
which was established before Independence,
the growth of the company was incremental year by year,
but due to online selling of books and wide spread Internet access
its annual growth started to collapse,seeing sharp downfalls, 
you as a Data Scientist help this heritage book store 
gain its popularity back and increase footfall of customers 
and provide ways the business can improve exponentially, apply Association 
RuleAlgorithm, explain the rules, and visualize the graphs for clear understanding of solution.

What is the business objective?
The main objective is to create a association rules to recommend relevant 
books to seller based on support,confidence and lift.
In addition to the Association rules Model prediction, we also have taken 
into account the  recommendation for a sale increase to a book seller.

Are there any constraints?
Understanding the metric for evaluation was a challenge as well.
Since the data consisted of binary data, EDA of  was a major challenge.
Lastly we can not find which books are sold most

"""

import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
#Let us import the book data set
books=pd.read_csv("C:/Data Set/book.csv")
books.head()
frequent_itemsets=apriori(books,min_support=0.03,max_len=4,use_colnames=True)
#items or item set must have minimum support value 3 % sale i.e.0.03
#you will get support values for 1,2,3 and 4 max items

frequent_itemsets.sort_values('support',ascending=False,inplace=True)
#Support values will be sorted in descending order
#If we will  frequent_itemsets,43 % sale is of cookbook,
#42 % sale is of child book
#28 % sale is of Do it yourself book
#27.6 % sale is of geography book

rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1) 
#This generate association rules of size 1198X9 columns comprizes of antescends,consequences
# let us generate books having  mini.3 % sale
rules1=rules[rules['support']==0.03]
rules1
#The books which are sold 3 % minimum in single or in combination but with confidence=0.7,0.8,0.9 and 1
rules2=rules1[rules1['confidence']==0.7]
rules2
#The books with confidence 70% genereate 0 rules
rules2=rules1[rules1['confidence']==0.8]
rules2
#The books with confidence 80% genereate 0 rules
#similarly with 90% it generates 0 rules
#let us try with 100% confidence
rules2=rules1[rules1['confidence']==1]
rules2
#It generates two rules
#1 if customer is buying DoItYBks, ItalArt then there is 100 % chances it will buy Artbook
# which is lowest saling title,you can provide 20% discount for the same
#2 if cutomer is buying ChildBks, CookBks, ItalArt then there are 100 % chances that cutomer will buy artbook
#offer 20% discount to enhance the sale

# Now let us try for min support value 0.04
frequent_itemsets=apriori(books,min_support=0.02,max_len=4,use_colnames=True)
frequent_itemsets.sort_values('support',ascending=False,inplace=True)
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1) 

rules1=rules[rules['support']==0.02]
rules1
#The books which are sold 3 % minimum in single or in combination but with confidence=0.7,0.8,0.9 and 1
rules2=rules1[rules1['confidence']==1]
rules2

# here also artbk and refbook are having less sale
frequent_itemsets=apriori(books,min_support=0.05,max_len=4,use_colnames=True)
frequent_itemsets.sort_values('support',ascending=False,inplace=True)
frequent_itemsets
