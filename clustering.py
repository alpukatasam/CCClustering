import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.cluster import KMeans

model = pickle.load(open('CC GENERAL.sav', 'rb'))

df=pd.read_excel("CCClust.xlsx")
features = ['BALANCE', 'PURCHASES']
X = df[features]

st.title('CREDIT CARD CLUSTERING')

numClusters = st.slider("Pilih Jumlah Cluster", min_value=2, max_value=8, value=4)

model = KMeans(n_clusters=numClusters)
clusters = model.fit_predict(X)

fig, ax = plt.subplots()
scatter = ax.scatter(X['BALANCE'], X['PURCHASES'], c=clusters, cmap='viridis')
ax.set_xlabel('BALANCE')
ax.set_ylabel('PURCHASES')


legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

st.pyplot(fig)
