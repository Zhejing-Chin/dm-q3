import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import awesome_streamlit as ast

from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

def write():


	with st.spinner("Loading classifier ..."):
		# ast.shared.components.title_awesome("- Classification")
		st.title('Classification')
		st.info('This page is for you to checkout the training resukt of each classifier!')

		'''
		### Top K features based on RFECV
		'''

		X_fs = pd.read_csv('dataset/SelectedFeatures_dataframe.csv')

		st.text('\n\nTotal Number of Top Features based on RFECV are {}'.format(X_fs.shape[1]))


		#---------------------------------------------------
		### Classification 

		st.info('DecisionTreeClassifier max_depth fine tuning')
		st.image('images/dt.png', use_column_width=True)

		st.info('RandomForestClassifier max_depth fine tuning')
		st.image('images/rf 1.png', use_column_width=True)

		st.info('RandomForestClassifier n_estimators fine tuning')
		st.image('images/rf 2.png', use_column_width=True)

		st.info('GradientBoostingClassifier max_depth fine tuning')
		st.image('images/gb 1.png', use_column_width=True)

		st.info('GradientBoostingClassifier n_estimators fine tuning')
		st.image('images/gb 2.png', use_column_width=True)

		st.info('DecisionTreeClassifier')
		st.image(['images/dt-s.png', 'images/dt-ns.png'], 
			caption=["SMOTE", "No SMOTE"], 
			width=300)
		
		st.info('RandomForestClassifier')
		st.image(['images/rf-s.png', 'images/rf-ns.png'], 
			caption=["SMOTE", "No SMOTE"], 
			width=300)

		st.info('GradientBoostingClassifier')
		st.image(['images/gb-s.png', 'images/gb-ns.png'], 
			caption=["SMOTE", "No SMOTE"], 
			width=300)

		st.image('images/roc.png', use_column_width=True)















		
		