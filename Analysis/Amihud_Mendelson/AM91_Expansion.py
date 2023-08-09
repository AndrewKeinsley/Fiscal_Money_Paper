import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm


dataframe_raw = pd.read_excel('Treasury_Data.xlsx')

##### CLEANING THE DATA #####

# Narrowing the dataframe to only Bonds, Notes, and Bills, respectively.
df_narrowed = dataframe_raw.loc[dataframe_raw['itype'].isin([1,2,4])]

# Adjusting certain variables
df_narrowed[['tmyld']] = df_narrowed[['tmyld']]*36500

# Reducing the number of variables for ease of use
df = df_narrowed[['crspid','tcusip','mcaldt','tmatdt','tcouprt','itype','tmbid','tmask','tmaccint','tmduratn','tmyld']]

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

# Years to Maturity
df['y2mat'] = ((df.tmatdt - df.mcaldt)/np.timedelta64(1, 'Y'))
df['y2mat'] = df['y2mat'].astype(int)

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
notes['d2mat_check_Note'] = notes['d2mat']
bonds = df.loc[df['itype'] == 1]

notes = notes.sort_values(['d2mat'], ascending=[True])
bills = bills.sort_values(['d2mat'], ascending=[True])
bonds = bonds.sort_values(['d2mat'], ascending=[True])

##### MATCHING SECURITIES BY DAYS TO MATURITY #####
bonds = pd.merge_asof(bonds, notes, on='d2mat', by='mcaldt', direction='nearest', tolerance=1, suffixes=('_Bond', '_Note'))

bonds = bonds.sort_values(['mcaldt','d2mat'], ascending=[True,True])

bonds_matched = bonds.dropna()

notes = pd.merge_asof(notes, bills, on='d2mat', by='mcaldt', direction='nearest', tolerance=1, suffixes=('_Note', '_Bill'))

notes = notes.sort_values(['mcaldt','d2mat'], ascending=[True,True])

notes_matched = notes.dropna()

notes_matched = notes_matched.loc[notes_matched['m2mat_Note'] < 7]

##### CALCULATING RELATIVE RATES (NOTES-BILLS)#####

notes_matched['tmyld_Spread'] = notes_matched['tmyld_Note'] - notes_matched['tmyld_Bill']
notes_matched['baspread_Spread'] = notes_matched['baspread_Note'] - notes_matched['baspread_Bill']
notes_matched['tcouprt_Spread'] = notes_matched['tcouprt_Note'] - notes_matched['tcouprt_Bill']
notes_matched['d2mat_Spread'] = notes_matched['d2mat']-notes_matched['d2mat_check']
notes_matched['year'] = notes_matched['mcaldt'].dt.year

##### REGRESSION ANALYSIS (NOTES-BILLS) #####

Y = notes_matched['tmyld_Spread']
X = notes_matched[['baspread_Spread','d2mat_Spread','tcouprt_Spread','m2mat_Note','year']]

X = sm.add_constant(X,prepend=True)

model = sm.OLS(Y,X,)
results = model.fit(cov_type='HC1')

print('\n\n ****** NOTES - BILLS REGRESSION ****** \n\n')
print(results.summary())

# print('\n\n ****** LaTeX VERSION ****** \n\n')
# print(results.summary().as_latex())



##### CALCULATING RELATIVE RATES (BONDS-NOTES) #####

bonds_matched['tmyld_Spread'] = bonds_matched['tmyld_Bond'] - bonds_matched['tmyld_Note']
bonds_matched['baspread_Spread'] = bonds_matched['baspread_Bond'] - bonds_matched['baspread_Note']
bonds_matched['tcouprt_Spread'] = bonds_matched['tcouprt_Bond'] - bonds_matched['tcouprt_Note']
bonds_matched['d2mat_Spread'] = bonds_matched['d2mat']-bonds_matched['d2mat_check_Note']
bonds_matched['year'] = bonds_matched['mcaldt'].dt.year

##### REGRESSION ANALYSIS (BONDS-NOTES) #####

Y = bonds_matched['tmyld_Spread']
X = bonds_matched[['baspread_Spread','d2mat_Spread','tcouprt_Spread','y2mat_Bond','year']]

X = sm.add_constant(X,prepend=True)

model = sm.OLS(Y,X,)
results = model.fit(cov_type='HC1')

print('\n\n ****** BONDS - NOTES REGRESSION ****** \n\n')
print(results.summary())

# print('\n\n ****** LaTeX VERSION ****** \n\n')
# print(results.summary().as_latex())