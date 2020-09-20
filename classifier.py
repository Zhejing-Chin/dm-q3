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
		st.header('Classification')

		'''
		### Select Top K features based on RFECV
		'''
		bank_ = pd.read_csv('bank_.csv')

		X = bank_.drop(['Decision'], 1)
		X = X.astype(object)
		X = pd.get_dummies(X)

		st.dataframe(X)

		y = bank_[['Decision']]
		y = y.replace({"accept": 1, "reject": 0})
		y = y.iloc[:, 0]

		from imblearn.over_sampling import SMOTE 

		sm = SMOTE(random_state = 2, sampling_strategy="minority", k_neighbors=5) 
		X_res, y_res = sm.fit_sample(X, y)


		X_fs = pd.read_csv('SelectedFeatures_dataframe.csv')

		st.text('\n\nTotal Number of Top Features based on RFECV are {}}'.format(X_fs.shape[1]))


		#---------------------------------------------------
		### Classification - Training

		X_train, X_test, y_train, y_test = train_test_split(X_fs, y_res, test_size=0.3, random_state=1)


		max_depth = np.linspace(1, 32, 32, endpoint=True)
		train_results = []
		test_results = []
		for depth in max_depth:
		   rf = DecisionTreeClassifier(random_state=1, max_depth=depth, min_samples_leaf=1)
		   rf.fit(X_train, y_train)
		   train_pred = rf.predict(X_train)
		   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
		   roc_auc = auc(false_positive_rate, true_positive_rate)
		   train_results.append(roc_auc)
		   y_pred = rf.predict(X_test)
		   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
		   roc_auc = auc(false_positive_rate, true_positive_rate)
		   test_results.append(roc_auc)


		from matplotlib.legend_handler import HandlerLine2D
		line1, = plt.plot(max_depth, train_results, 'b', label="Train AUC")
		line2, = plt.plot(max_depth, test_results, 'r', label="Test AUC")
		plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
		plt.ylabel('AUC score')
		plt.xlabel('max_depth')
		plt.title('DecisionTreeClassifier max_depth fine tuning')
		st.pyplot()


		max_depth = np.linspace(1, 32, 32, endpoint=True)
		train_results = []
		test_results = []
		for depth in max_depth:
		   rf = RandomForestClassifier(n_estimators=20, n_jobs=-1, random_state=1, class_weight="balanced", max_depth=depth)
		   rf.fit(X_train, y_train)
		   train_pred = rf.predict(X_train)
		   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
		   roc_auc = auc(false_positive_rate, true_positive_rate)
		   train_results.append(roc_auc)
		   y_pred = rf.predict(X_test)
		   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
		   roc_auc = auc(false_positive_rate, true_positive_rate)
		   test_results.append(roc_auc)
		    
		line1, = plt.plot(max_depth, train_results, 'b', label="Train AUC")
		line2, = plt.plot(max_depth, test_results, 'r', label="Test AUC")
		plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
		plt.ylabel('AUC score')
		plt.xlabel('max_depth')
		plt.title('RandomForestClassifier max_depth fine tuning')
		st.pyplot()


		n_estimators = [1, 2, 4, 8, 16, 18, 20, 25]
		train_results = []
		test_results = []
		for estimator in n_estimators:
		   rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1, random_state=1, class_weight="balanced", max_depth=10)
		   rf.fit(X_train, y_train)
		   train_pred = rf.predict(X_train)
		   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
		   roc_auc = auc(false_positive_rate, true_positive_rate)
		   train_results.append(roc_auc)
		   y_pred = rf.predict(X_test)
		   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
		   roc_auc = auc(false_positive_rate, true_positive_rate)
		   test_results.append(roc_auc)


		line1, = plt.plot(n_estimators, train_results, 'b', label="Train AUC")
		line2, = plt.plot(n_estimators, test_results, 'r', label="Test AUC")
		plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
		plt.ylabel('AUC score')
		plt.xlabel('n_estimators')
		plt.title('RandomForestClassifier n_estimators fine tuning')
		st.pyplot()


		max_depth = np.linspace(1, 32, 32, endpoint=True)
		train_results = []
		test_results = []
		for depth in max_depth:
		   rf = GradientBoostingClassifier(random_state=1, n_estimators=32, 
		                                   learning_rate=0.1, max_depth=depth, max_features='auto')
		   rf.fit(X_train, y_train)
		   train_pred = rf.predict(X_train)
		   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
		   roc_auc = auc(false_positive_rate, true_positive_rate)
		   train_results.append(roc_auc)
		   y_pred = rf.predict(X_test)
		   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
		   roc_auc = auc(false_positive_rate, true_positive_rate)
		   test_results.append(roc_auc)


		line1, = plt.plot(max_depth, train_results, 'b', label="Train AUC")
		line2, = plt.plot(max_depth, test_results, 'r', label="Test AUC")
		plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
		plt.ylabel('AUC score')
		plt.xlabel('max_depth')
		plt.title('GradientBoostingClassifier max_depth fine tuning')
		st.pyplot()


		n_estimators = [1, 2, 4, 8, 16, 18, 20, 25, 30, 50, 100]
		train_results = []
		test_results = []
		for estimator in n_estimators:
		   rf = GradientBoostingClassifier(random_state=1, n_estimators=estimator, 
		                                   learning_rate=0.1, max_depth=5, max_features='auto')
		   rf.fit(X_train, y_train)
		   train_pred = rf.predict(X_train)
		   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
		   roc_auc = auc(false_positive_rate, true_positive_rate)
		   train_results.append(roc_auc)
		   y_pred = rf.predict(X_test)
		   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
		   roc_auc = auc(false_positive_rate, true_positive_rate)
		   test_results.append(roc_auc)


		line1, = plt.plot(n_estimators, train_results, 'b', label="Train AUC")
		line2, = plt.plot(n_estimators, test_results, 'r', label="Test AUC")
		plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
		plt.ylabel('AUC score')
		plt.xlabel('n_estimators')
		plt.title('GradientBoostingClassifier n_estimators fine tuning')
		st.pyplot()

		'''
		### SMOTEd Classification
		'''

		fig, axes = plt.subplots(1, 2, figsize=(13,6))


		# Create different classifiers.
		classifiers = {
		    'DecisionTreeClassifier': [DecisionTreeClassifier(random_state=1, max_depth=20, 
		                                                      max_features ='auto'), 'orange'],
		    'RandomForestClassifier': [RandomForestClassifier(n_jobs=-1, class_weight="balanced", 
		                                                      max_depth = 10, n_estimators = 10, random_state=1), 'blue'],
		    'GradientBoostingClassifier': [GradientBoostingClassifier(random_state=1, n_estimators=20, 
		                                                             learning_rate=0.1, max_depth = 5, max_features='auto'), 'red'],
		}

		n_classifiers = len(classifiers)

		for index, (name, (classifier, color)) in enumerate(classifiers.items()):
		    classifier.fit(X_train, y_train)
		    y_pred = classifier.predict(X_test)
		    prob_clf = classifier.predict_proba(X_test)
		    prob_clf = prob_clf[:, 1]
		    auc_clf = roc_auc_score(y_test, prob_clf)

		    st.text("-----{}----- \n".format(name))
		    st.text("Accuracy on training set : {:.2f}".format(classifier.score(X_train, y_train)))
		    st.text("Accuracy on test set     : {:.2f}".format(classifier.score(X_test, y_test)))
		    st.text('AUC: %.2f' % auc_clf)

		    confusion_majority=confusion_matrix(y_test, y_pred)
		#     st.write('\nMajority classifier Confusion Matrix\n', confusion_majority)
		    
		    st.text('\n**********************')
		    st.text('Majority TN = '.format( confusion_majority[0][0]))
		    st.text('Majority FP = '.format( confusion_majority[0][1]))
		    st.text('Majority FN = '.format( confusion_majority[1][0]))
		    st.text('Majority TP = '.format( confusion_majority[1][1]))
		    st.text('**********************\n')

		    st.text('\nPrecision= {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
		    st.text('Recall= {:.2f}'. format(recall_score(y_test, y_pred, average='weighted')))
		    st.text('F1= {:.2f}'. format(f1_score(y_test, y_pred, average='weighted')))
		    st.text('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))
		    
		    fpr_clf, tpr_clf, thresholds_clf = roc_curve(y_test, prob_clf) 
		    axes[0].plot(fpr_clf, tpr_clf, color=color, label=name + ' AUC = %0.2f'% auc_clf) 
		    st.text("\n")
		    
		    
		axes[0].plot([0, 1], [0, 1], "g--")
		axes[0].set_xlabel('False Positive Rate')
		axes[0].set_ylabel('True Positive Rate')
		axes[0].set_title('SMOTE - Receiver Operating Characteristic (ROC) Curve')
		# st.pyplot()

		'''
		### No SMOTE Classification
		'''
		topFeaturesRFE = np.array(X_fs.columns)
		NS_X_fs = X[topFeaturesRFE]

		ns_X_train, ns_X_test, ns_y_train, ns_y_test = train_test_split(NS_X_fs, y, test_size=0.3, random_state=1)

		# Create different classifiers.
		classifiers = {
		    'DecisionTreeClassifier': [DecisionTreeClassifier(random_state=1, max_depth=20, 
		                                                      max_features ='auto'), 'orange'],
		    'RandomForestClassifier': [RandomForestClassifier(n_jobs=-1, class_weight="balanced", 
		                                                      max_depth = 10, n_estimators = 10, random_state=1), 'blue'],
		    'GradientBoostingClassifier': [GradientBoostingClassifier(random_state=1, n_estimators=20, 
		                                                             learning_rate=0.1, max_depth = 5, max_features='auto'), 'red'],
		}

		n_classifiers = len(classifiers)
		for index, (name, (classifier, color)) in enumerate(classifiers.items()):
		    classifier.fit(ns_X_train, ns_y_train)
		    y_pred = classifier.predict(ns_X_test)
		    prob_clf = classifier.predict_proba(ns_X_test)
		    prob_clf = prob_clf[:, 1]
		    auc_clf = roc_auc_score(ns_y_test, prob_clf)

		    st.text("-----{}----- \n".format(name))
		    st.text("Accuracy on training set : {:.2f}".format(classifier.score(ns_X_train, ns_y_train)))
		    st.text("Accuracy on test set     : {:.2f}".format(classifier.score(ns_X_test, ns_y_test)))
		    st.text('AUC: %.2f' % auc_clf)

		    confusion_majority=confusion_matrix(ns_y_test, y_pred)
		#     st.text('\nMajority classifier Confusion Matrix\n', confusion_majority)
		    
		    st.text('\n**********************')
		    st.text('Majority TN = '.format( confusion_majority[0][0]))
		    st.text('Majority FP = '.format( confusion_majority[0][1]))
		    st.text('Majority FN = '.format( confusion_majority[1][0]))
		    st.text('Majority TP = '.format( confusion_majority[1][1]))
		    st.text('**********************\n')

		    st.text('\nPrecision= {:.2f}'.format(precision_score(ns_y_test, y_pred, average='weighted')))
		    st.text('Recall= {:.2f}'. format(recall_score(ns_y_test, y_pred, average='weighted')))
		    st.text('F1= {:.2f}'. format(f1_score(ns_y_test, y_pred, average='weighted')))
		    st.text('Accuracy= {:.2f}'. format(accuracy_score(ns_y_test, y_pred)))
		    
		    fpr_clf, tpr_clf, thresholds_clf = roc_curve(ns_y_test, prob_clf) 
		    axes[1].plot(fpr_clf, tpr_clf, color=color, label=name + ' AUC = %0.2f'% auc_clf) 
		    st.text("\n")
		    
		    
		axes[1].set_xlabel('False Positive Rate')
		axes[1].set_ylabel('True Positive Rate')
		axes[1].set_title('No SMOTE - Receiver Operating Characteristic (ROC) Curve')
		axes[1].plot([0, 1], [0, 1], color='green', linestyle='--')


		st.pyplot(fig)
