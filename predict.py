import streamlit as st
import awesome_streamlit as ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def write():

	with st.spinner("Loading prediction ..."):
		# ast.shared.components.title_awesome("- Prediction")

		st.header('Prediction')

		X_fs = pd.read_csv('SelectedFeatures_dataframe.csv')
		topFeaturesRFE = np.array(X_fs.columns)


		'''
		### Predict - with Pre-trained Model

		'''

		Employment_Type = st.selectbox(
			"Choose your employment type.",
			['employer', 'self_employed', 'government', 
		     'employee', 'fresh_graduate'],
		)

		More_Than_One_Products = st.selectbox(
			"Do you have more than one bank product?",
			['yes', 'no'],
		)

		Credit_Card_types= st.selectbox(
			"Choose your credit card type.",
			['platinum', 'normal', 'gold'],
		)

		Property_Type= st.selectbox(
			"Choose your property type.",
			['condominium', 'bungalow', 'terrace', 'flat'],
		)

		State= st.selectbox(
			"Choose a your state.",
			['Johor', 'Selangor', 'Kuala Lumpur', 'Penang', 'Negeri Sembilan', 
		         'Sarawak', 'Sabah', 'Terengganu', 'Kedah'],
		)

		Credit_Card_Exceed_Months = st.slider('Credit card exceed months', min_value=1, max_value=7, step=1)
		Loan_Tenure_Year = st.slider('Loan tenure year', min_value=10.0, max_value=24.0, step=1.0)
		Number_of_Dependents = st.slider('Number of dependents', min_value=2, max_value=6, step=1)
		Years_to_Financial_Freedom = st.slider('Year to financial freedom', min_value=5.0, max_value=19.0, step=1.0)
		Number_of_Credit_Card_Facility = st.slider('Number of credit card facility', min_value=2.0, max_value=6.0, step=1.0)
		Number_of_Properties = st.slider('Number of properties', min_value=2.0, max_value=5.0, step=1.0)
		Number_of_Bank_Products = st.slider('Number of bank products', min_value=1.0, max_value=5.0, step=1.0)
		Number_of_Loan_to_Approve = st.slider('Number of loan to approve', min_value=1, max_value=3, step=1)
		Years_for_Property_to_Completion = st.slider('Years for property to completion', min_value=10.0, max_value=13.0, step=1.0)
		Number_of_Side_Income = st.slider('Number of side income', min_value=1.0, max_value=3.0, step=1.0)

		Loan_Amount = st.slider('Loan amount', min_value=0.0, max_value=1000000.0, step=1.0)
		Monthly_Salary = st.slider('Monthly salary', min_value=0.0, max_value=100000.0, step=1.0)
		Total_Sum_of_Loan = st.slider('Total sum of loan',min_value=0.0, max_value=10000000.0, step=1.0)
		Total_Income_for_Join_Application = st.slider('Total income for join application', min_value=0.0, max_value=100000.0, step=1.0)
		Score = st.slider('Score ', min_value=6, max_value=9, step=1)

		bank = pd.read_csv('bank.csv')

		new_X_test = bank.drop(['Decision'], 1)

		new_input = np.array([Credit_Card_Exceed_Months, Employment_Type, Loan_Amount,
		       Loan_Tenure_Year, More_Than_One_Products, Credit_Card_types,
		       Number_of_Dependents, Years_to_Financial_Freedom,
		       Number_of_Credit_Card_Facility, Number_of_Properties,
		       Number_of_Bank_Products, Number_of_Loan_to_Approve, Property_Type,
		       Years_for_Property_to_Completion, State, Number_of_Side_Income,
		       Monthly_Salary, Total_Sum_of_Loan,
		       Total_Income_for_Join_Application, Score])

		new = np.array(pd.get_dummies(pd.DataFrame([new_input], columns=new_X_test.columns)).columns)

		for i in new:
		    new_x = list(map(lambda x: 1 if (i == x) else 0, topFeaturesRFE))
		    
		new_x = np.array(new_x)

		input_X_test = pd.DataFrame([new_x], columns=topFeaturesRFE)


		classifier_user = st.selectbox(
			"Choose desire classifier.",
			['DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier'],
		)

		'''
		### Predict Result: 
		'''
		import joblib

		if st.button('Predict!'):
			loaded_model = joblib.load(open(classifier_user +".sav", 'rb'))
			result = loaded_model.predict(input_X_test)
			st.text(list(map(lambda x: 'Accept' if (x == 1) else 'Reject', result))[0])

