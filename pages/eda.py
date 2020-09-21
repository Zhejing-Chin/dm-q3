import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import awesome_streamlit as ast


def write():

	with st.spinner("Loading EDA ..."):
		# ast.shared.components.title_awesome("- Exploratory Data Analysis")

		st.title('Exploratory Data Analysis')

		'''
		# DM Project - 
		## Question 3: Bank Loan Decision
		### Read Data - Bank

		'''
		st.info('The data')
		bank = pd.read_csv('dataset/Bank_CS.csv')
		st.dataframe(bank)

		'''
		### Clean Data (Clean inconsistency, Fill nulls)
		'''


		bank = bank.drop(['Unnamed: 0', 'Unnamed: 0.1'], 1)
		st.info('Describing the data')
		st.dataframe(bank.describe())

		nulls = bank.columns[bank.isna().any()].tolist()

		st.info('How many percent it contains N/A?')
		for i,col in enumerate(nulls):
		    st.text('Percent of missing data for ' + col + ' : ' + str((bank[col].isnull().sum()/bank.shape[0])*100) + '%' )
		    

		num_cols = bank[['Loan_Amount','Monthly_Salary', 'Total_Sum_of_Loan', 'Total_Income_for_Join_Application']]

		'''
		#### Boxplot of numerical data
		'''

		st.info('Are there any outliers?')
		fig,axes = plt.subplots(2,2,figsize=(20,10))
		for idx,numCols in enumerate(num_cols):
		    row,col = divmod(idx,2)
		    sns.boxplot(x=numCols,data=bank,y='Decision',ax=axes[row,col], palette="Blues")
		#print(numDf.describe())
		plt.tight_layout()  
		st.pyplot()


		cat_cols = bank.select_dtypes(include=['object']).copy()

		for col in cat_cols.columns:
			bank[col] = bank[col].str.lower()
			if col == "State":
				bank['State'] = bank['State'].replace(dict.fromkeys(['johor','johor b'], 'Johor'))
				bank['State'] = bank['State'].replace(dict.fromkeys(['selangor'], 'Selangor'))
				bank['State'] = bank['State'].replace(dict.fromkeys(['kuala lumpur','k.l'], 'Kuala Lumpur'))
				bank['State'] = bank['State'].replace(dict.fromkeys(['penang','p.pinang','pulau penang'], 'Penang'))
				bank['State'] = bank['State'].replace(dict.fromkeys(['n.sembilan','n.s'], 'Negeri Sembilan'))
				bank['State'] = bank['State'].replace(dict.fromkeys(['sarawak','swk'], 'Sarawak'))
				bank['State'] = bank['State'].replace(dict.fromkeys(['sabah'], 'Sabah'))
				bank['State'] = bank['State'].replace(dict.fromkeys(['kedah'], 'Kedah'))
				bank['State'] = bank['State'].replace(dict.fromkeys(['trengganu'], 'Terengganu'))
			# st.text(col + ' : '  + str(bank[col].unique()))


		cnum_cols = bank[['Loan_Amount','Monthly_Salary', 'Total_Sum_of_Loan', 'Total_Income_for_Join_Application']]
		cat_cols = bank.drop(num_cols.columns, 1)

		for i in cat_cols.columns:
		    bank[i].fillna(bank[i].value_counts().index[0] , inplace = True)
		    bank[i] = bank[i].astype(object)
		    
		for i in num_cols.columns:
		    bank[i].fillna(bank[i].mean(), inplace = True) 
		    
		bank = bank.drop_duplicates()

		'''
		### Null values are gone!! 
		'''
		st.info('Cleaned!')
		st.write(bank.isna().sum())

		'''
		### Transform Data (Binning, Encode)

		'''

		from sklearn.preprocessing import LabelEncoder

		bank_ = bank.copy()

		num_cols = bank.select_dtypes(include = ['int64', 'float64'])

		for i in num_cols:
		    bank_[i] = pd.cut(bank_[i], bins=3, precision=0, duplicates='drop')
		    
		bank_[num_cols.columns] = bank_[num_cols.columns].apply(LabelEncoder().fit_transform)
		bank_[num_cols.columns] = bank_[num_cols.columns].replace({0: "Low", 1: "Medium", 2: "High"})

		'''
		### We binned the numerical data 
		'''
		st.info('We categorized the data for better visualization!')
		st.write(bank_.astype('object'))


		'''
		## EDA

		'''

		bank_eda = bank_.copy()

		# Rearrange dataframe
		X_eda = bank_eda.drop(['Decision'], 1)
		y_eda = bank_eda['Decision']

		bank_eda = pd.concat([X_eda, y_eda], 1)


		genre = st.selectbox(
			"Which descriptive graph would you like to check?",
			options=bank_eda.columns,
		)

		fig, axes = plt.subplots(figsize = (10, 5))

		if st.button('Done!'):
			if (genre == 'State'):
			    g = sns.countplot(x = genre, data = bank_eda, ax = axes, orient = 'v', palette = 'rocket')
			    plt.setp(g.get_xticklabels(), rotation=90)
			else:
			    g = sns.countplot(x = genre, data = bank_eda, ax = axes, orient = 'v', palette = 'rocket')

			for p in g.patches:
			        height = p.get_height()
			        axes.text(p.get_x() + p.get_width()/2,
			                height + 3,
			                '{:1.0f}'.format(height),
			                ha="center") 

			plt.tight_layout()
			st.pyplot()

		'''
		### Heatmap
		'''
		from sklearn.preprocessing import normalize

		num_cols = bank[['Credit_Card_Exceed_Months', 'Loan_Amount', 'Monthly_Salary', 'Total_Sum_of_Loan', 'Total_Income_for_Join_Application',
		                    'Loan_Tenure_Year', 'Number_of_Dependents', 'Years_to_Financial_Freedom', 'Number_of_Credit_Card_Facility', 'Number_of_Properties', 
		                    'Number_of_Bank_Products', 'Number_of_Loan_to_Approve', 'Years_for_Property_to_Completion', 'Number_of_Side_Income',
		                    'Score']]



		bank_eda_norm = normalize(num_cols)
		bank_eda_norm = pd.DataFrame(bank_eda_norm, columns = num_cols.columns)

		st.info('The relations between the variables.')
		# Finding the relations between the variables.
		plt.figure(figsize=(10,10))
		c = abs(bank_eda_norm.corr())
		sns.heatmap(c, cmap="YlOrBr", annot=True)
		st.pyplot()



		questions = [["The distributions of Property Loan based on Employment Type.", 0], 
		["The distribution of Join Income based on customers' Score.", 1], 
		["The distribution of Join Income in each State.", 2],
		["The distribution of Credit Card Type based on Monthly Salary.", 3],
		["Which State tend to have higher Sum of Loan?", 4], 
		["Which type of Credit Card holder tend to have more Loan to approve?", 5]]

		df_questions = pd.DataFrame(questions, columns=['Questions', 'ID'])
		values = df_questions['Questions'].tolist()
		options = df_questions['ID'].tolist()
		dic = dict(zip(options, values))

		genre = st.selectbox(
			"Choose a question.",
			options,
			format_func=lambda x: dic[x],
		)

		fig, ax = plt.subplots(figsize=(10,7))

		if st.button('OK!'):
			if genre == 0: 
				g = sns.countplot(data=bank_eda,x='Employment_Type',
					hue='Property_Type', ax=ax, palette='mako')
			elif genre == 1:
				g = sns.countplot(data=bank_eda, x='Score',
					hue='Total_Income_for_Join_Application', ax=ax, palette='mako')
			elif genre == 4:
				g = sns.countplot(data=bank_eda, x='State',
					hue='Total_Sum_of_Loan', ax=ax, palette='mako')
			elif genre == 5:
				g = sns.countplot(data=bank_eda, x='Credit_Card_types',
					hue='Number_of_Loan_to_Approve', ax=ax, palette='mako')
			elif genre == 2:
				g = sns.countplot(data=bank_eda, x='State',
					hue='Total_Income_for_Join_Application', ax=ax, palette='mako')
			else: 
				g = sns.countplot(data=bank_eda, x='Monthly_Salary',
					hue='Credit_Card_types', ax=ax, palette='mako')

			for p in g.patches:
				height = p.get_height()
				ax.text(p.get_x() + p.get_width()/2,
					height + 3,
					'{:1.0f}'.format(height),
					ha="center")

			st.pyplot()


		'''
		### Decision is Imbalance
		'''
		st.info('The acceptance rate is imabalance.')
		st.title("Imbalance Data")

		sns.countplot(x = "Decision", palette="Paired", data=bank)
		st.pyplot()

