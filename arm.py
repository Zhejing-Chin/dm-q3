import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import awesome_streamlit as ast

def write():

	with st.spinner("Loading ARM ..."):
		# ast.shared.components.title_awesome("- Association Rules Mining")
		st.header('Association Rules Mining')

		'''
		### Association Rule Mining
		'''

		from mlxtend.frequent_patterns import apriori, association_rules


		bank_arm = pd.read_csv('bank_arm.csv')

		frequent_itemsets = apriori(bank_arm, min_support=0.1, use_colnames=True)

		rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1) 
		rules = rules.sort_values(['support', 'confidence', 'lift'], ascending = [False, False, False]) 

		st.dataframe(rules)


