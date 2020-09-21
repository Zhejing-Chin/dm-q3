import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import awesome_streamlit as ast

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


def write():

	with st.spinner("Loading feature selection ..."):
		# ast.shared.components.title_awesome("- Feature Selection")

		st.title('Feature Selection')


		'''
		### Feature Selection - Boruta all Features

		'''
		st.header('Boruta')
		st.image('images/boruta all.png', use_column_width=True)

		st.image('images/boruta acc-features.png', use_column_width=True)



		'''
		### Feature Selection - RFECV all Features

		'''
		st.header('RFECV')
		st.image('images/rfecv all.png', use_column_width=True)
		st.image('images/rfecv acc-features.png', use_column_width=True)

		'''
		### Feature Selection - Chi2 all Features

		'''
		st.header('Chi2')
		st.image('images/chi2 all.png', use_column_width=True)
		st.image('images/chi2 acc-features.png', use_column_width=True)









