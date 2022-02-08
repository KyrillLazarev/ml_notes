import os
import json
import pprint
import pandas as pd
#####################################
os.getcwd()
files = []
names = os.listdir(os.getcwd() + "\\data")
for name in names:
    fullname = os.path.join(os.getcwd(), name)
    files.append(fullname.split('\\')[-1])
files
####################################
with open("data//", encoding='utf-8') as f:
    data = json.load(f)
for feature in data['feature']:
    pprint.pprint(feature[['propertie']])
####################################
dataframes = []
for file in files:
    with open(f"data//{file}", encoding='utf-8') as f:
        data = json.load(f)
    df = pd.json_normalize(data['features'])
    dataframes.append(df)
data_final = pd.concat(dataframes, ignore_index=True)
####################################
pd.set_option('display.max_columns', 22)
data_final.head()
