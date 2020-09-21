import streamlit as st
import awesome_streamlit as ast
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import awesome_streamlit as ast

def write():


	with st.spinner("Loading Clustering ..."):
		# ast.shared.components.title_awesome("")
		st.title('Clustering')
		'''
		### Clustering
		'''
		import seaborn as sns
	
		from sklearn.preprocessing import MinMaxScaler
		from sklearn.preprocessing import normalize

		bank_cluster = pd.read_csv('dataset/bank.csv')

		num_cols = bank_cluster[['Loan_Amount','Monthly_Salary', 'Total_Sum_of_Loan', 'Total_Income_for_Join_Application']]

		X_cluster = normalize(num_cols)
		X_cluster = pd.DataFrame(X_cluster, columns=num_cols.columns)

		'''
		### Clustering - AgglomerativeClustering
		'''
		st.header("Agglomerative Clustering")
		from sklearn.cluster import AgglomerativeClustering

		st.info('The optimal N for Agglomerative Clustering.')
		st.image('images/aggC dendro.png', use_column_width=True)

		st.info('Clustering all numerical data')
		st.image('images/aggc.png', use_column_width=True)

		st.info('Clustering X,  y numerical data')
		n=3
		agglomerative = AgglomerativeClustering(n_clusters = n, affinity='euclidean', linkage='ward')
		agglomerative.fit_predict(X_cluster)

		agg_X = st.selectbox(
			"Which X would you like to check?",
			options=X_cluster.columns,
		)

		agg_y = st.selectbox(
			"Which y would you like to check?",
			options=X_cluster.columns,
		)

		fig, axes = plt.subplots(figsize=(13,6))
		sns.scatterplot(x=agg_X, y=agg_y, hue=agglomerative.labels_, data=X_cluster, ax=axes)
		st.pyplot()

		'''
		### Clustering - KMeans
		'''
		st.header("K-Means Clustering")
		from sklearn.cluster import KMeans

		# KMeans clustering

		
		st.info('Optimal N for K-Means')
		st.image('images/Kmeans elbow.png', use_column_width=True)

		n = 3
		km = KMeans(
		        n_clusters = n, 
		        init = 'random',
		        n_init = 10, 
		        max_iter = 300, 
		        tol = 1e-04, 
		        random_state = 1
		    )
		km.fit(X_cluster)

		st.info('Clustering X, y numerical data')
		kmeans_X = st.selectbox(
			"Which numerical data, X,  would you like to check?",
			options=X_cluster.columns,
		)

		kmeans_y = st.selectbox(
			"Which numerical data, y,  would you like to check?",
			options=X_cluster.columns,
		)
		fig, axes = plt.subplots(figsize=(13,6))
		sns.scatterplot(x=kmeans_X, y=kmeans_y, hue=km.labels_, data=X_cluster, ax=axes)

		st.pyplot()


		'''
		### Clustering - Silhouette
		'''
		st.header("Silhouette Score")
		# Silhouette

		from sklearn.metrics import silhouette_score 
		from yellowbrick.cluster import silhouette_visualizer
		from sklearn.cluster import KMeans 

		st.text("Silhouette Score (n = {}) = {}".format(n, silhouette_score(X_cluster, km.labels_)))
		st.image('images/Silhouette.png', use_column_width=True)

		'''
		### Clustering - K-Modes
		'''
		from kmodes.kmodes import KModes
		from sklearn import preprocessing

		st.header("K-Modes Clustering")

		st.info('Optimal K')
		st.image('images/KModes choose K.png', use_column_width=True)
		le = preprocessing.LabelEncoder()
		bank_le = bank_cluster.select_dtypes(include=['object'])
		bank_le = bank_le.apply(le.fit_transform)

		km_cao = KModes(n_clusters=7, init = "Cao", n_init = 1, verbose=1)
		fitClusters_cao = km_cao.fit_predict(bank_le)

		st.info('Distribution of predicted cluster')
		st.image('images/KModes cluster predicted.png', use_column_width=True)

		bank_le = bank_cluster.reset_index()
		clustersDf = pd.DataFrame(fitClusters_cao)
		clustersDf.columns = ['cluster_predicted']
		combinedDf = pd.concat([bank_le, clustersDf], axis = 1).reset_index()
		combinedDf = combinedDf.drop(['index', 'level_0'], axis = 1)

		st.info('Clustering X categorical data')
		plt.subplots(figsize = (15,5))

		kmodes_X = st.selectbox(
			"Which categorical data would you like to check?",
			options=combinedDf.select_dtypes(include=['object', 'int64']).columns,
		)

		sns.countplot(x=combinedDf[kmodes_X], order=combinedDf[kmodes_X].value_counts().index,hue=combinedDf['cluster_predicted'])
		st.pyplot()







