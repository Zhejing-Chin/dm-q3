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
		ast.shared.components.title_awesome("")
		'''
		### Clustering
		'''

		import seaborn as sns

		
		from sklearn.preprocessing import MinMaxScaler
		from sklearn.preprocessing import normalize



		bank_cluster = pd.read_csv('bank.csv')

		num_cols = bank_cluster[['Loan_Amount','Monthly_Salary', 'Total_Sum_of_Loan', 'Total_Income_for_Join_Application']]

		X_cluster = normalize(num_cols)
		X_cluster = pd.DataFrame(X_cluster, columns=num_cols.columns)

		'''
		### Clustering - DBSCAN
		'''

		from sklearn.cluster import DBSCAN

		dbscan = DBSCAN(eps=0.5, min_samples=20, algorithm='brute').fit(X_cluster)

		sns.relplot(x="Total_Sum_of_Loan", y="Monthly_Salary", hue=bank_cluster['Property_Type'], data=X_cluster)
		st.pyplot()

		'''
		### Clustering - AgglomerativeClustering
		'''

		from sklearn.cluster import AgglomerativeClustering

		n=4
		agglomerative = AgglomerativeClustering(n_clusters = n, affinity='euclidean', linkage='ward')
		agglomerative.fit_predict(X_cluster)
		sns.relplot(x="Loan_Amount", y="Monthly_Salary", hue=agglomerative.labels_, data=X_cluster)
		sns.relplot(x="Loan_Amount", y="Monthly_Salary", hue=bank_cluster['Property_Type'], data=X_cluster)
		st.pyplot()

		'''
		### Clustering - KMeans
		'''

		from sklearn.cluster import KMeans

		# KMeans clustering
		import seaborn as sns

		distortions = []
		N_range = 10
		# your codes here...
		for i in range(1, N_range):
		    km = KMeans(
		        n_clusters = i, 
		        init = 'k-means++', 
		        random_state = 4
		    )
		    km.fit(X_cluster)
		    distortions.append(km.inertia_)
		    

		# plot
		plt.subplots(figsize=(12,5))
		plt.subplot(121)
		plt.plot(range(1, N_range), distortions, marker='o')
		plt.xlabel('Number of clusters')
		plt.ylabel('Distortion')
		plt.title('Elbow Method')
		st.pyplot()

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

		fig, axes = plt.subplots(1, 2, figsize=(13,6))
		sns.scatterplot(x="Loan_Amount", y="Monthly_Salary", hue=bank_cluster['Score'], data=X_cluster, ax=axes[0])
		sns.scatterplot(x="Loan_Amount", y="Monthly_Salary", hue=km.labels_, data=X_cluster, ax=axes[1])

		st.pyplot()


		'''
		### Clustering - Silhouette
		'''

		# Silhouette

		from sklearn.metrics import silhouette_score 
		from yellowbrick.cluster import silhouette_visualizer
		from sklearn.cluster import KMeans 

		st.text("Silhoutte Score (n = {}) = {}".format(n, silhouette_score(X_cluster, km.labels_)))
		silhouette_visualizer(KMeans(n, random_state=12), X_cluster, colors='yellowbrick')
		st.pyplot()
