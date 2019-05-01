# %%
import os
import pandas as pd
path = './case_2_new/data/'
os.chdir(path)

file_list = [c for c in os.listdir('.') if c[-3:]=='csv']
# [c[:-7]+'aug.csv' for c in os.listdir('.') if c[-3:]=='csv']

files_dict = {}

for file in file_list:
    print('----> reading: ',file)
    files_dict[file] = pd.read_csv(file,nrows=20,encoding = "ISO-8859-1")
    print(files_dict[file].dtypes)
