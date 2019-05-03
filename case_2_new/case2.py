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


import pandas
import random

# loads header + s random lines from each file
s = 100 #desired sample size
for filename in file_list:
    print('----> counting lines in:',file)
    #number of records in file (excludes header)
    n = sum(1 for line in open(filename)) - 1
    print('---->',n,'lines counted, sampling',n,'lines')
    skip = sorted(random.sample(range(1,n+1),n-s))
    #the 0-indexed header will not be included in the skip list
    df = pandas.read_csv(filename, skiprows=skip, encoding = "ISO-8859-1")
    print(files_dict[file].dtypes)

for k,v in files_dict:
    print(k,v.shape)


# vector of visitors of 1/0s for customer/non customer
Y = []

# matrix of properties belonging to person
