import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import awesome_streamlit as ast
import eda
import arm
import cluster
import classifier
import fs
import predict

ast.core.services.other.set_logging_format()

PAGES = {
    "Exploratory Data Analysis": eda,
    "Association Rules Mining": arm,
    "Clustering": cluster,
    "Feature Selection": fs,
    "Classification": classifier,
    "Prediction": predict,

}


def main():


	'''
	
	'''
	import seaborn as sns
	sns.set()
	st.set_option('deprecation.showPyplotGlobalUse', False)

	st.sidebar.title("Navigation")
	selection = st.sidebar.radio("Go to", list(PAGES.keys()))

	page = PAGES[selection]

	with st.spinner(f"Loading {selection} ..."):
		ast.shared.components.write_page(page)


	


if __name__ == "__main__":
    main()













