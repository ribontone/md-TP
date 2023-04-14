
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
#import plotly.express as px
#from wordcloud import WordCloud, STOPWORDS
#from IPython.display import Image
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pycountry


import pyarrow as pa
import pyarrow.parquet as pq

import glob


#eplURL = './datasets/EPLMatches.csv'
#datasetGames=pd.read_csv(eplURL)
#datasetGames = datasetGames.drop(datasetGames.columns[0], axis=1)


datasets = {}
for filename in glob.glob('./datasets/20**-**.csv'):
    df = pd.read_csv(filename)
    key = filename.replace('.csv', '')
    key = key.replace('./datasets\\', '')
    datasets[key] = df


datasetPosicoes=pd.read_csv('./datasets/EPLStandings.csv')

# Calculate the number of missing values in each row
missing_values = datasetPosicoes.isnull().sum(axis=1)
# Sort the dataframe by the number of missing values in each row
df_sorted = datasetPosicoes.iloc[missing_values.argsort()]
# Reset the index of the sorted dataframe
df_sorted = df_sorted.reset_index(drop=True)

datasetPosicoes=df_sorted

del df,filename,key,df_sorted,missing_values





"""
print o nome das colunas


for name, df in datasets.items():
    print(f" {name}:")
    print(' '.join(df.columns))
    print()  


"""





