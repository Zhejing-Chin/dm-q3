import streamlit as st
import awesome_streamlit as ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def write():

	with st.spinner("Loading prediction ..."):
		# ast.shared.components.title_awesome("- Prediction")

		st.title('Prediction')
		st.info('This page is for you to make prediction')


		X_fs = pd.read_csv('dataset/SelectedFeatures_dataframe.csv')
		topFeaturesRFE = np.array(X_fs.columns)


		'''
		### Predict - with Pre-trained Model

		'''
		st.sidebar.title("Input for prediction")

		def user_input_features():
			Employment_Type = st.sidebar.selectbox(
				"Choose your employment type.",
				['employer', 'self_employed', 'government', 
			     'employee', 'fresh_graduate'],
			)

			More_Than_One_Products = st.sidebar.selectbox(
				"Do you have more than one bank product?",
				['yes', 'no'],
			)

			Credit_Card_types= st.sidebar.selectbox(
				"Choose your credit card type.",
				['platinum', 'normal', 'gold'],
			)

			Property_Type= st.sidebar.selectbox(
				"Choose your property type.",
				['condominium', 'bungalow', 'terrace', 'flat'],
			)

			State= st.sidebar.selectbox(
				"Choose a your state.",
				['Johor', 'Selangor', 'Kuala Lumpur', 'Penang', 'Negeri Sembilan', 
			         'Sarawak', 'Sabah', 'Terengganu', 'Kedah'],
			)

			Credit_Card_Exceed_Months = st.sidebar.slider('Credit card exceed months', min_value=1, max_value=7, step=1)
			Loan_Tenure_Year = st.sidebar.slider('Loan tenure year', min_value=10.0, max_value=24.0, step=1.0)
			Number_of_Dependents = st.sidebar.slider('Number of dependents', min_value=2, max_value=6, step=1)
			Years_to_Financial_Freedom = st.sidebar.slider('Year to financial freedom', min_value=5.0, max_value=19.0, step=1.0)
			Number_of_Credit_Card_Facility = st.sidebar.slider('Number of credit card facility', min_value=2.0, max_value=6.0, step=1.0)
			Number_of_Properties = st.sidebar.slider('Number of properties', min_value=2.0, max_value=5.0, step=1.0)
			Number_of_Bank_Products = st.sidebar.slider('Number of bank products', min_value=1.0, max_value=5.0, step=1.0)
			Number_of_Loan_to_Approve = st.sidebar.slider('Number of loan to approve', min_value=1, max_value=3, step=1)
			Years_for_Property_to_Completion = st.sidebar.slider('Years for property to completion', min_value=10.0, max_value=13.0, step=1.0)
			Number_of_Side_Income = st.sidebar.slider('Number of side income', min_value=1.0, max_value=3.0, step=1.0)

			Loan_Amount = st.sidebar.slider('Loan amount', min_value=0.0, max_value=1000000.0, step=1.0)
			Monthly_Salary = st.sidebar.slider('Monthly salary', min_value=0.0, max_value=100000.0, step=1.0)
			Total_Sum_of_Loan = st.sidebar.slider('Total sum of loan',min_value=0.0, max_value=10000000.0, step=1.0)
			Total_Income_for_Join_Application = st.sidebar.slider('Total income for join application', min_value=0.0, max_value=100000.0, step=1.0)
			Score = st.sidebar.slider('Score ', min_value=6, max_value=9, step=1)

			data = {
			   'Credit_Card_Exceed_Months': Credit_Card_Exceed_Months, 
			   'Employment_Type': Employment_Type, 
			   'Loan_Amount': Loan_Amount,
		       'Loan_Tenure_Year': Loan_Tenure_Year, 
		       'More_Than_One_Products': More_Than_One_Products, 
		       'Credit_Card_types': Credit_Card_types,
		       'Number_of_Dependents': Number_of_Dependents, 
		       'Years_to_Financial_Freedom': Years_to_Financial_Freedom,
		       'Number_of_Credit_Card_Facility': Number_of_Credit_Card_Facility, 
		       'Number_of_Properties': Number_of_Properties,
		       'Number_of_Bank_Products': Number_of_Bank_Products, 
		       'Number_of_Loan_to_Approve': Number_of_Loan_to_Approve, 
		       'Property_Type': Property_Type,
		       'Years_for_Property_to_Completion': 
		       'Years_for_Property_to_Completion', 
		       'State': State, 
		       'Number_of_Side_Income': Number_of_Side_Income,
		       'Monthly_Salary': Monthly_Salary, 
		       'Total_Sum_of_Loan': Total_Sum_of_Loan,
		       'Total_Income_for_Join_Application': Total_Income_for_Join_Application, 
		       'Score': Score

			}
			features = pd.DataFrame(data, index=[0])
			return features

		input_df = user_input_features()

		# bank = pd.read_csv('dataset/bank.csv')

		# new_X_test = bank.drop(['Decision'], 1)

		# new_input = np.array([Credit_Card_Exceed_Months, Employment_Type, Loan_Amount,
		#        Loan_Tenure_Year, More_Than_One_Products, Credit_Card_types,
		#        Number_of_Dependents, Years_to_Financial_Freedom,
		#        Number_of_Credit_Card_Facility, Number_of_Properties,
		#        Number_of_Bank_Products, Number_of_Loan_to_Approve, Property_Type,
		#        Years_for_Property_to_Completion, State, Number_of_Side_Income,
		#        Monthly_Salary, Total_Sum_of_Loan,
		#        Total_Income_for_Join_Application, Score])

		new = np.array(pd.get_dummies(input_df))

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
			loaded_model = joblib.load(open('pretrained_classifier/'+classifier_user +".sav", 'rb'))
			result = loaded_model.predict(input_X_test)
			result_proba = loaded_model.predict_proba(input_X_test)
			st.text("The probability for Reject is : {} ".format(result_proba[0][0]))
			st.text("The probability for Accept is : {} ".format(result_proba[0][1]))
			st.text("Hence, the predicted result is: {} ".format(list(map(lambda x: 'Accept' if (x == 1) else 'Reject', result))[0]))
