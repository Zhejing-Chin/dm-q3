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

		st.header('Feature Selection')

		bank = pd.read_csv('Bank_CS.csv')
		bank = bank.drop(['Unnamed: 0', 'Unnamed: 0.1'], 1)

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


		num_cols = bank[['Loan_Amount','Monthly_Salary', 'Total_Sum_of_Loan', 'Total_Income_for_Join_Application']]
		cat_cols = bank.drop(num_cols.columns, 1)

		for i in cat_cols.columns:
		    bank[i].fillna(bank[i].value_counts().index[0] , inplace = True)
		    bank[i] = bank[i].astype(object)
		    
		for i in num_cols.columns:
		    bank[i].fillna(bank[i].mean(), inplace = True) 
		    
		bank = bank.drop_duplicates()

		from sklearn.preprocessing import LabelEncoder

		bank_ = bank.copy()

		num_cols = bank.select_dtypes(include = ['int64', 'float64'])

		for i in num_cols:
		    bank_[i] = pd.cut(bank_[i], bins=3, precision=0, duplicates='drop')
		    
		bank_[num_cols.columns] = bank_[num_cols.columns].apply(LabelEncoder().fit_transform)
		bank_[num_cols.columns] = bank_[num_cols.columns].replace({0: "Low", 1: "Medium", 2: "High"})





		### Generate X, y

		X = bank_.drop(['Decision'], 1)
		X = pd.get_dummies(X)

		y = bank_[['Decision']]
		y = y.replace({"accept": 1, "reject": 0})
		y = y.iloc[:, 0]

		### SMOTE X, y

		from imblearn.over_sampling import SMOTE 

		sm = SMOTE(random_state = 2, sampling_strategy="minority", k_neighbors=5) 
		X_res, y_res = sm.fit_sample(X, y)


		'''
		### Feature Selection - Boruta all Features

		'''
		from boruta import BorutaPy
		from sklearn.preprocessing import MinMaxScaler

		def ranking(ranks, names, order=1):
		    minmax = MinMaxScaler()
		    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
		    ranks = map(lambda x: round(x,2), ranks)
		    return dict(zip(names, ranks))


		rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth = 5)

		feat_selector = BorutaPy(rf, n_estimators = "auto", random_state = 1)


		feat_selector.fit(X.values, y.values)
		colnames = X.columns
		boruta_score = ranking(list(map(float, feat_selector.ranking_)), colnames, order=-1)
		boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])
		boruta_score_sort = boruta_score.sort_values('Score', ascending = False)


		sns.catplot(x="Score", y="Features", data = boruta_score_sort[:], kind = "bar", 
		               height=14, aspect=1.9, palette='coolwarm')
		plt.title("Boruta all Features: Decision")
		st.pyplot()

		from collections import defaultdict

		accuracies_iteration = defaultdict(list)
		for i in range(1,boruta_score.shape[0]):
		    colnames = boruta_score.head(i)['Features'].tolist()
		    X_clf = X_res[colnames]
		    X_train, X_test, y_train, y_test = train_test_split(X_clf,y_res,test_size=0.3, random_state=0)
		    rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth = 13, n_estimators = 20, random_state=1)
		    rf.fit(X_train,y_train)
		    y_pred=rf.predict(X_test)
		    accuracies_iteration[i].append(rf.score(X_test, y_test))

		result = pd.DataFrame.from_dict(accuracies_iteration)
		plt.figure()
		plt.xlabel("Number of features selected")
		plt.ylabel("Accuracy Score")
		plt.title('BORUTA')
		plt.plot(result.columns, result.values.ravel())
		st.pyplot()




		'''
		### Feature Selection - RFECV all Features

		'''

		from sklearn.feature_selection import RFECV

		rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth = 5, n_estimators = 100)

		rf.fit(X, y)
		rfe = RFECV(rf, min_features_to_select = 1, cv = 5)
		rfe.fit(X, y)

		colnames = X.columns
		rfe_score = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
		rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
		rfe_score = rfe_score.sort_values("Score", ascending = False)

		sns.catplot(x="Score", y="Features", data = rfe_score, kind = "bar", 
		            height=14, aspect=1.9, palette='coolwarm')
		st.pyplot()


		accuracies_iteration = defaultdict(list)
		for i in range(1,rfe_score.shape[0]):
		    colnames = rfe_score.head(i)['Features'].tolist()
		    X_clf = X_res[colnames]
		    X_train, X_test, y_train, y_test = train_test_split(X_clf,y_res,test_size=0.3, random_state=0)
		    rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth = 13, n_estimators = 20, random_state=1)
		    rf.fit(X_train,y_train)
		    y_pred=rf.predict(X_test)
		    accuracies_iteration[i].append(rf.score(X_test, y_test))

		result = pd.DataFrame.from_dict(accuracies_iteration)
		plt.figure()
		plt.xlabel("Number of features selected")
		plt.ylabel("Accuracy Score")
		plt.title('RFECV')
		plt.plot(result.columns, result.values.ravel())
		st.pyplot()


		'''
		### Feature Selection - Chi2 all Features

		'''

		from sklearn.feature_selection import SelectKBest
		from sklearn.feature_selection import chi2

		bestfeatures = SelectKBest(score_func=chi2, k=10)
		fit = bestfeatures.fit(X,y)
		dfscores = pd.DataFrame(fit.scores_)
		dfcolumns = pd.DataFrame(X.columns)

		Chi2_Scores = pd.concat([dfcolumns, dfscores],axis=1)
		Chi2_Scores.columns = ['Features','Score']  
		Chi2_Scores = Chi2_Scores.sort_values("Score", ascending = False)

		sns.catplot(x="Score", y="Features", data = Chi2_Scores, kind = "bar", 
		               height=14, aspect=1.9, palette='coolwarm')
		plt.title("Chi2 all Features: Decision")
		st.pyplot()



		accuracies_iteration = defaultdict(list)
		for i in range(1,Chi2_Scores.shape[0]):
		    colnames = Chi2_Scores.head(i)['Features'].tolist()
		    X_clf = X_res[colnames]
		    X_train, X_test, y_train, y_test = train_test_split(X_clf,y_res,test_size=0.3, random_state=0)
		    rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth = 13, n_estimators = 20, random_state=1)
		    rf.fit(X_train,y_train)
		    y_pred=rf.predict(X_test)
		    accuracies_iteration[i].append(rf.score(X_test, y_test))

		# print("Accuracy on test set: {:.3f}".format(clf.score(X_test, y_test)))
		result = pd.DataFrame.from_dict(accuracies_iteration)
		plt.figure()
		plt.xlabel("Number of features selected")
		plt.ylabel("Accuracy Score")
		plt.title('Chi2')
		plt.plot(result.columns, result.values.ravel())
		st.pyplot()










