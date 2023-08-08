import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


dataframe_raw = pd.read_excel('Treasury_Data.xlsx')

##### CLEANING THE DATA #####

# Narrowing the dataframe to only Bonds, Notes, and Bills, respectively.
df_narrowed = dataframe_raw.loc[dataframe_raw['itype'].isin([1,2,4])]

# Reducing the number of variables for ease of use
df = df_narrowed[['crspid','tcusip','mcaldt','tmatdt','tcouprt','itype','tmbid','tmask','tmaccint','tmduratn','tmpcyld']]

# Dropping any observations with NaN values
df = df.dropna()

# Dropping negative values from the numeric columns
numvals = df.select_dtypes(include=['float64'])
posvals = (numvals[numvals.columns] >= 0).all(axis=1)
df = df[posvals].reset_index(drop=True)

# Checking the data for anymore issues
df.info()
df.describe()

##### CALCULATING NEEDED STATISTICS #####

# Relative Bid-Ask Spread (Amihud and Mendelson, 1991; Section 2)

df['baspread'] = (df.tmask-df.tmbid)/(df.tmask+df.tmaccint)

# Months to Maturity
df['m2mat'] = ((df.tmatdt - df.mcaldt)/np.timedelta64(1, 'M'))
df['m2mat'] = df['m2mat'].astype(int)

# Days to Maturity
df['d2mat'] = ((df.tmatdt - df.mcaldt)/np.timedelta64(1, 'D'))
df['d2mat'] = df['d2mat'].astype(int)

##### SEPARATING BY TYPE #####

bills = df.loc[df['itype'] == 4]
bills['d2mat_check'] = bills['d2mat']
notes = df.loc[df['itype'] == 2]
bonds = df.loc[df['itype'] == 1]

notes = notes.sort_values(['d2mat'], ascending=[True])
bills = bills.sort_values(['d2mat'], ascending=[True])

##### MATCHING BILLS TO NOTES #####
notes = pd.merge_asof(notes, bills, on='d2mat', by='mcaldt', direction='nearest', tolerance=3,suffixes=('_Note', '_Bill'))

notes = notes.sort_values(['mcaldt','d2mat'], ascending=[True,True])

notes_matched = notes.dropna()

##### CALCULATING RELATIVE RATES #####

notes_matched['tmpcyld_Spread'] = notes_matched['tmpcyld_Note'] - notes_matched['tmpcyld_Bill']
notes_matched['baspread_Spread'] = notes_matched['baspread_Note'] - notes_matched['baspread_Bill']
notes_matched['tcouprt_Spread'] = notes_matched['tcouprt_Note'] - notes_matched['tcouprt_Bill']
