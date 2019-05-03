# %%
import os
import pandas as pd
import random
path = './case_2_new/data/'
os.chdir(path)

file_list = [c for c in os.listdir('.') if c[-3:]=='csv']
# [c[:-7]+'aug.csv' for c in os.listdir('.') if c[-3:]=='csv']

# %%
files_dict = {}

# loads header + s random lines from each file
s = 1000 #desired sample size
e = "ISO-8859-1"
randomize = 1
for filename in file_list:
    print('---->',filename)
    if randomize:
        loaded_successfully = 0
        while not loaded_successfully:
            try:
                print('-> counting lines')
                # number of records in file (excludes header)
                n = sum(1 for line in open(filename,encoding=e)) - 1
                print('->',n,'lines counted, sampling',s,'lines')
                skip = sorted(random.sample(range(1,n+1),n-s))
                #the 0-indexed header will not be included in the skip list
                df = pd.read_csv(filename, skiprows=skip, encoding = e)
                files_dict[filename] = df
                loaded_successfully = 1
            except ParserError:
                print('ParserError Caught: Error reading. Randomizing sample')
    else:
        df = pd.read_csv(filename,nrows=s,encoding =e)
        files_dict[filename] = df
        print('->',df.shape)

    # print(files_dict[file].dtypes)
# %%
# build session dictionary
u_sess_set = set(files_dict['sessionstart_aug.csv']['sessionnumber'].unique())
# u_sess_set = set()
unique_sess_dict = {}
for k,v in files_dict.items():
    a = v['sessionnumber'].unique()
    # unique_sess_dict[k] = set(a)
    print(len(u_sess_set))
    u_sess_set = u_sess_set.intersection(a)

# %%
u_sess_set

files_dict['sessionstart_aug.csv']['sessionnumber'].nunique()
files_dict['sessionsummary_aug.csv']['sessionnumber'].nunique()
files_dict['sessionstart_aug.csv']['sessionnumber'].nunique()
# vector of visitors of 1/0s for customer/non customer
Y = []

# matrix of properties belonging to person
