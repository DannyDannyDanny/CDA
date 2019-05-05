# %%
import os
import pandas as pd
import random
path = './case_2_new/data/'
os.chdir(path)

file_list = [c for c in os.listdir('.') if c[-3:]=='csv']
# [c[:-7]+'aug.csv' for c in os.listdir('.') if c[-3:]=='csv']
# %%
s = 10000 #desired sample size
e = "ISO-8859-1"
df_vis = pd.read_csv("visitor_aug.csv",
                 nrows=s,
                 encoding =e,
                 error_bad_lines = False)
offset = 200
n_each = 200
list_cust0 = df_vis[df_vis.iscustomer==1].sessionnumber.unique()[offset:offset+n_each]
list_cust1 = df_vis[df_vis.iscustomer==0].sessionnumber.unique()[offset:offset+n_each]
list_mix = list_cust0+list_cust1

# for filename in *.csv
# filename = 'page_aug.csv'
# iter_csv = pd.read_csv(filename,
#                        encoding = e,
#                        error_bad_lines = False,
#                        iterator=True,
#                        chunksize=1000)
# df = pd.concat([ch[ch['sessionnumber'].isin(list_cust0)] for ch in iter_csv])


# %%
file_col_dict = {}
for filename in file_list:
    df = pd.read_csv(filename,nrows=2,encoding =e,error_bad_lines = False)
    file_col_dict[filename] = list(df.columns)

# %%
removecols = 0
if removecols:
    # file_col_dict['visitor_aug.csv']
    file_col_dict['visitor_aug.csv'].remove(sessionnumber)
    file_col_dict['visitor_aug.csv'].remove(uvtisnewacquisition)
    file_col_dict['visitor_aug.csv'].remove(uvtacquisitiontimestamp)
    file_col_dict['visitor_aug.csv'].remove(iscustomer)
    # file_col_dict['in_out_of_eBank_aug.csv']
    file_col_dict['in_out_of_eBank_aug.csv'].remove(sessionnumber)
    file_col_dict['in_out_of_eBank_aug.csv'].remove(pagesequenceinsession)
    file_col_dict['in_out_of_eBank_aug.csv'].remove(ebank_page)
    # file_col_dict['pagesummary_aug.csv']
    file_col_dict['pagesummary_aug.csv'].remove(sessionnumber)
    file_col_dict['pagesummary_aug.csv'].remove(pageinstanceid)
    file_col_dict['pagesummary_aug.csv'].remove(pageloadduration)
    file_col_dict['pagesummary_aug.csv'].remove(pageimagesloadduration)
    file_col_dict['pagesummary_aug.csv'].remove(pageviewtime)
    file_col_dict['pagesummary_aug.csv'].remove(pageviewactivetime)
    file_col_dict['pagesummary_aug.csv'].remove(pagescrollmaxdistance)
    file_col_dict['pagesummary_aug.csv'].remove(pagepopupallowed)
    file_col_dict['pagesummary_aug.csv'].remove(pagepopupblocked)
    file_col_dict['pagesummary_aug.csv'].remove(windowwidth)
    file_col_dict['pagesummary_aug.csv'].remove(windowheight)
    file_col_dict['pagesummary_aug.csv'].remove(eventtimestamp)
    # file_col_dict['sessionsummary_aug.csv']
    file_col_dict['sessionsummary_aug.csv'].remove(sessionnumber)
    file_col_dict['sessionsummary_aug.csv'].remove(devicelanguage)
    file_col_dict['sessionsummary_aug.csv'].remove(deviceflashversion)
    file_col_dict['sessionsummary_aug.csv'].remove(deviceconnectiontype)
    file_col_dict['sessionsummary_aug.csv'].remove(devicescreencolourdepth)
    file_col_dict['sessionsummary_aug.csv'].remove(devicescreenresolutionwidth)
    file_col_dict['sessionsummary_aug.csv'].remove(devicescreenresolutionheight)
    file_col_dict['sessionsummary_aug.csv'].remove(sessionfullcollection)
    file_col_dict['sessionsummary_aug.csv'].remove(sessionpopupallowed)
    file_col_dict['sessionsummary_aug.csv'].remove(sessionpopupblocked)
    file_col_dict['sessionsummary_aug.csv'].remove(sessionnetworkspeeddatasize)
    file_col_dict['sessionsummary_aug.csv'].remove(sessionnetworkspeeddatatime)
    file_col_dict['sessionsummary_aug.csv'].remove(sessioncontentbytes)
    file_col_dict['sessionsummary_aug.csv'].remove(sessioncontenttransmittime)
    file_col_dict['sessionsummary_aug.csv'].remove(timezoneoffset)
    file_col_dict['sessionsummary_aug.csv'].remove(eventtimestamp)
    file_col_dict['sessionsummary_aug.csv'].remove(sessionviewtime)
    file_col_dict['sessionsummary_aug.csv'].remove(sessionviewactivetime)
    file_col_dict['sessionsummary_aug.csv'].remove(sessionviewidletime)
    file_col_dict['sessionsummary_aug.csv'].remove(sessionviewoffsitetime)
    file_col_dict['sessionsummary_aug.csv'].remove(campaignattributioncount)
    file_col_dict['sessionsummary_aug.csv'].remove(lasteventtimestamp)
    file_col_dict['sessionsummary_aug.csv'].remove(lastclientlocaltimestamp)
    file_col_dict['sessionsummary_aug.csv'].remove(lastactiveeventtimestamp)
    file_col_dict['sessionsummary_aug.csv'].remove(sessionscripterrors)
    file_col_dict['sessionsummary_aug.csv'].remove(sessionmetricvalue)
    file_col_dict['sessionsummary_aug.csv'].remove(sessionpagescount)
    # file_col_dict['page_aug.csv']
    file_col_dict['page_aug.csv'].remove(sessionnumber)
    file_col_dict['page_aug.csv'].remove(pagelocationdomain)
    file_col_dict['page_aug.csv'].remove(pagetitle)
    file_col_dict['page_aug.csv'].remove(referringpageinstanceid)
    file_col_dict['page_aug.csv'].remove(pageinstanceid)
    file_col_dict['page_aug.csv'].remove(eventtimestamp)
    file_col_dict['page_aug.csv'].remove(attributionsequenceinsession)
    file_col_dict['page_aug.csv'].remove(pagesequenceinsession)
    file_col_dict['page_aug.csv'].remove(pagesequenceinattribution)
    file_col_dict['page_aug.csv'].remove(iscustomer)
    file_col_dict['page_aug.csv'].remove(pagelocation)
    # file_col_dict['sessionstart_aug.csv']
    file_col_dict['sessionstart_aug.csv'].remove(sessionnumber)
    file_col_dict['sessionstart_aug.csv'].remove(deviceuseragent)
    file_col_dict['sessionstart_aug.csv'].remove(devicebrowsername)
    file_col_dict['sessionstart_aug.csv'].remove(devicebrowserversion)
    file_col_dict['sessionstart_aug.csv'].remove(devicecookiesenabled)
    file_col_dict['sessionstart_aug.csv'].remove(deviceplatformname)
    file_col_dict['sessionstart_aug.csv'].remove(deviceplatformgroup)
    file_col_dict['sessionstart_aug.csv'].remove(sessiondatasourcename)
    file_col_dict['sessionstart_aug.csv'].remove(devicesystemname)
    file_col_dict['sessionstart_aug.csv'].remove(devicesystemtype)
    file_col_dict['sessionstart_aug.csv'].remove(sessionpartialoptout)
    file_col_dict['sessionstart_aug.csv'].remove(sessionconfiguredtimeout)
    file_col_dict['sessionstart_aug.csv'].remove(sessioniscontinuation)
    file_col_dict['sessionstart_aug.csv'].remove(linkedsessionnumber)
    file_col_dict['sessionstart_aug.csv'].remove(linkedsessionkey)
    file_col_dict['sessionstart_aug.csv'].remove(eventtimestamp)
    file_col_dict['sessionstart_aug.csv'].remove(sessionbasecurrencycode)
    # file_col_dict['click_aug.csv']
    file_col_dict['click_aug.csv'].remove(sessionnumber)
    file_col_dict['click_aug.csv'].remove(pageinstanceid)
    file_col_dict['click_aug.csv'].remove(objectalttext)
    file_col_dict['click_aug.csv'].remove(unformattedhierarchyname)
    file_col_dict['click_aug.csv'].remove(unformattedhierarchyid)
    file_col_dict['click_aug.csv'].remove(objecthierarchyname)
    file_col_dict['click_aug.csv'].remove(objecthierarchyid)
    file_col_dict['click_aug.csv'].remove(objectsrc)
    file_col_dict['click_aug.csv'].remove(objectislink)
    file_col_dict['click_aug.csv'].remove(objectisdownload)
    file_col_dict['click_aug.csv'].remove(objecttype)
    file_col_dict['click_aug.csv'].remove(objectsourcetype)
    file_col_dict['click_aug.csv'].remove(objecttagname)
    file_col_dict['click_aug.csv'].remove(fieldischaineditemselect)
    file_col_dict['click_aug.csv'].remove(previouschainediseventindex)
    file_col_dict['click_aug.csv'].remove(formname)
    file_col_dict['click_aug.csv'].remove(formid)
    file_col_dict['click_aug.csv'].remove(objecthref)
    file_col_dict['click_aug.csv'].remove(objectid)
    file_col_dict['click_aug.csv'].remove(objectclass)
    file_col_dict['click_aug.csv'].remove(objectname)
    file_col_dict['click_aug.csv'].remove(eventtimestamp)
    file_col_dict['click_aug.csv'].remove(pageeventindex)
else:
    print(0)
# %%
file_df_dict = {}
for filename in file_list:
    print(filename)
    iter_csv = pd.read_csv(filename,
                           # usecols=file_col_dict[filename],
                           encoding = e,
                           error_bad_lines = False,
                           iterator=True,
                           chunksize=1000)
    df = pd.concat([ch[ch['sessionnumber'].isin(list_mix)] for ch in iter_csv])
    print(df.shape)
    file_df_dict[filename] = df

file_df_dict
# %%
for filename,df in file_df_dict.items():
    print(filename)
    # dropcols = []
    # df.sessionnumber.nunique()
    print('unique ses. numbers:\t',df.sessionnumber.nunique())
    print('ses. numbers:\t\t',df.shape[0])
    print(8*'--')


# Descriptive statistics for each column
df.describe()
df = pd.get_dummies(df)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
subsets = train_test_split(features, labels, test_size = 0.25, random_state = 42)
train_features, test_features, train_labels, test_labels = subsets

# %%
print(22*'-')
!say done
# %%
df_vis.iscustomer.copy()
df_vis.columns
df_vis.dtypes

df1 = pd.DataFrame()

zz
df_vis.drop(columns=dropcols)
ser = pd.Series()

for

#%%
dropcols = ['uvtfrequency',
            'uvtprevioussessiontimestamp',
            'uvtprevioussessionnumber',
            # 'uvtisnewacquisition',
            'idsequencenumber',
            'eventtimestamp']
#%%


df_vis.drop(columns=dropcols)


# %%
for filename in file_list:
    break
    print('---->',filename)

    df = pd.read_csv(filename,
                     skiprows=skip,
                     encoding = e,
                     error_bad_lines = False)
    files_dict[filename] = df


# %%
files_dict = {}

# loads header + s random lines from each file
s = 1000000 #desired sample size
e = "ISO-8859-1"
randomize = 0
# for filename in sorted(file_list)[2:]:
for filename in file_list:
    print('---->',filename)
    if randomize:
        print('-> counting lines')
        # number of records in file (excludes header)
        n = sum(1 for line in open(filename,encoding=e)) - 1
        print('->',n,'lines counted, sampling',s,'lines')
        skip = sorted(random.sample(range(1,n+1),n-s))
        #the 0-indexed header will not be included in the skip list
        n
        df = pd.read_csv(filename,
                         skiprows=skip,
                         encoding = e,
                         error_bad_lines = False)

        files_dict[filename] = df
        loaded_successfully = 1
    else:
        df = pd.read_csv(filename,
                         nrows=s,
                         encoding =e,
                         error_bad_lines = False)
        files_dict[filename] = df
        print('->',df.shape)

    # print(files_dict[file].dtypes)
# %%
# build session dictionary
a = set(files_dict['sessionsummary_aug.csv']['sessionnumber'].unique())
b = set(files_dict['sessionstart_aug.csv']['sessionnumber'].unique())
len(a)+len(b)
len(a.union(b))
len(a | b)
a.intersection(b)
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
