import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import awesome_streamlit as ast
import pages.eda
import pages.arm
import pages.cluster
import pages.classifier
import pages.fs
import pages.predict

ast.core.services.other.set_logging_format()

PAGES = {
	"Prediction": pages.predict,
    "Exploratory Data Analysis": pages.eda,
    "Association Rules Mining": pages.arm,
    "Clustering": pages.cluster,
    "Feature Selection": pages.fs,
    "Classification": pages.classifier,
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













