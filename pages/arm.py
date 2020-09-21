import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import awesome_streamlit as ast

def write():

	with st.spinner("Loading ARM ..."):
		# ast.shared.components.title_awesome("- Association Rules Mining")
		st.title('Association Rules Mining')
		st.info('This page is showing the top 200 associated rules with apriori algorithm, sort according to their support, confidence, and lift level. '
			)

		'''
		### Association Rule Mining
		'''

		rules_200 = pd.read_csv('dataset/rules(200).csv')

		st.dataframe(rules_200)


