# importing packages
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
import os, json


parser = ArgumentParser()
parser.add_argument("-d", "--rootDir", dest="rootDir", help="Root Directory for the extraction process to begin")
parser.add_argument("-o", "--out", dest="outPath", help="Output Path for the extracted CSV")
args = parser.parse_args()
rootDir = args.rootDir

outPath = args.outPath

json_files=[]

for path in Path(rootDir).iterdir():
    if path.is_dir():
        json_files.extend([str(path)+'/'+pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')])

# for file in json_files:
#     df = pd.read_json(file)
#     print(df)
# use pandas.concat method
df = pd.concat([pd.DataFrame([pd.read_json(f_name,typ='series')]) for f_name in json_files])

# view the concatenated dataframe
# print(df)
print(df.head())
print(df.columns)
# print(df.iloc[:2,0])
# print(df.iloc[:2,1])
# print(df.iloc[:2,2])
# print(df.iloc[:2,3])
# print(df.iloc[:2,4])
# print(df.iloc[:2,5])
# df.rename(columns = {'src_fm_fc_ms_ff':'source'}, inplace = True)
# df.rename(columns = {'s':'source'}, inplace = True)

# convert dataframe to csv file
df.to_csv(outPath,index=False)

# load the resultant csv file
result = pd.read_csv(outPath)

# and view the data
print(result)
