import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm
import pandas_datareader.data as web

##### IMPORTING DATA #####

# Treasury Data
dataframe_raw = pd.read_excel('Treasury_Data.xlsx')

# Additional Time Series Data
start = datetime.datetime(1950,1,1)
end = datetime.datetime.now()
fred_data = web.DataReader(['T10Y2YM'], "fred", start, end)

##### CLEANING THE DATA #####

# Narrowing the dataframe to only Bonds, Notes, and Bills, respectively.
df_narrowed = dataframe_raw.loc[dataframe_raw['itype'].isin([1,2,4])]

# Adjusting certain variables
df_narrowed[['tmyld']] = df_narrowed[['tmyld']]*36500
    # Note: The ytm = tmyld*365*100 comes from CRSP's derivation of their tmytm variable, not the author.

# Reducing the number of variables for ease of use
df = df_narrowed[['crspid','tcusip','mcaldt','tmatdt','tcouprt','itype','tmbid','tmask','tmaccint','tmduratn','tmyld']]

# Dropping any observations with NaN values
df = df.dropna()

#### PLEASE READ #####
# From the CRSP documentation: 
#       "At the start of the day, October 16, 1996, our source for daily and monthly [...] chaged to GovPX."
#       "The ask price was determined by the FRBNY based on what they expect a typical bid-ask spread to be. The rule used
#           to make this derivation was not public domain"
#       This is a source of problems with the regression analyses, so I'm only running the post-1996 regressions.
df = df[~(df['mcaldt']<'1996-10-01')]

# Dropping negative values from the numeric columns
numvals = df.select_dtypes(include=['float64'])
posvals = (numvals[numvals.columns] >= 0).all(axis=1)
df = df[posvals].reset_index(drop=True)

# Merging Dataframes
df = df.sort_values(['mcaldt'], ascending=[True])

df = pd.merge_asof(df, fred_data, left_on='mcaldt', right_on='DATE', direction='nearest', allow_exact_matches=True)

# Checking the data for anymore issues
df.info()
df.describe()

##### CALCULATING NEEDED STATISTICS #####

# Relative Bid-Ask Spread (Amihud and Mendelson, 1991; Section 2) -- in percentage terms
    # Random Note: for some reason, the full formula doesn't work if the numerator equals 1, 
    # So we do it piece by piece
df['price_paid'] = df['tmask']+df['tmaccint']
df['ask_ratio'] = df['tmask']/df['price_paid']
df['bid_ratio'] = df['tmbid']/df['price_paid']
df['bidask_ratio']=df['ask_ratio']-df['bid_ratio']
df['bidask_ratio'] = df['bidask_ratio']*100

# Years to Maturity
df['y2mat'] = ((df.tmatdt - df.mcaldt)//np.timedelta64(1, 'Y'))

# Months to Maturity
df['m2mat'] = ((df.tmatdt - df.mcaldt)//np.timedelta64(1, 'M'))

# Days to Maturity
df['d2mat'] = ((df.tmatdt - df.mcaldt)//np.timedelta64(1, 'D'))

##### SEPARATING BY TYPE #####

bills = df.loc[df['itype'] == 4]
notes = df.loc[df['itype'] == 2]
bonds = df.loc[df['itype'] == 1]

notes = notes.sort_values(['d2mat'], ascending=[True])
bills = bills.sort_values(['d2mat'], ascending=[True])
bonds = bonds.sort_values(['d2mat'], ascending=[True])
bb = bonds

##### MATCHING SECURITIES BY DAYS TO MATURITY #####
bonds = pd.merge_asof(bonds, notes, on='d2mat', by='mcaldt', direction='nearest', tolerance=1, suffixes=('_Bond', '_Note'), allow_exact_matches=True)

bonds = bonds.sort_values(['mcaldt','d2mat'], ascending=[True,True])

bonds_matched = bonds.dropna()

notes = pd.merge_asof(notes, bills, on='d2mat', by='mcaldt', direction='nearest', tolerance=1, suffixes=('_Note', '_Bill'), allow_exact_matches=True)

notes = notes.sort_values(['mcaldt','d2mat'], ascending=[True,True])

notes_matched = notes.dropna()

notes_matched = notes_matched.loc[notes_matched['m2mat_Note'] < 7]

bb_matched = pd.merge_asof(bb, bills, on='d2mat', by='mcaldt', direction='nearest', tolerance=1, suffixes=('_Bond', '_Bill'), allow_exact_matches=True)

bb_matched = bb_matched.dropna()

bb_matched = bb_matched.loc[bb_matched['m2mat_Bond'] < 7]

##### CALCULATING RELATIVE RATES (NOTES-BILLS)#####

notes_matched['tmyld_Spread'] = notes_matched['tmyld_Note'] - notes_matched['tmyld_Bill']
notes_matched['bidask_ratio_Spread'] = notes_matched['bidask_ratio_Note'] - notes_matched['bidask_ratio_Bill']
notes_matched['tcouprt_Spread'] = notes_matched['tcouprt_Note'] - notes_matched['tcouprt_Bill']
notes_matched['d2mat_Spread'] = notes_matched['tmatdt_Note']-notes_matched['tmatdt_Bill']
notes_matched['d2mat_Spread'] = notes_matched['d2mat_Spread'] // np.timedelta64(1, "D")
notes_matched['year'] = notes_matched['mcaldt'].dt.year

##### REGRESSION ANALYSIS (NOTES-BILLS) #####

Y = notes_matched['tmyld_Spread']
X = notes_matched[['bidask_ratio_Spread','tcouprt_Spread','m2mat_Note','T10Y2YM_Bill']]
# X = notes_matched[['bidask_ratio_Spread','tcouprt_Spread','T10Y2YM_Bill']]

X = sm.add_constant(X,prepend=True)

model = sm.OLS(Y,X,)
results = model.fit(cov_type='HC1')

print('\n\n ****** NOTES - BILLS REGRESSION ****** \n\n')
print(results.summary())

# print('\n\n ****** LaTeX VERSION ****** \n\n')
# print(results.summary().as_latex())



##### CALCULATING RELATIVE RATES (BONDS-NOTES) #####

bonds_matched['tmyld_Spread'] = bonds_matched['tmyld_Bond'] - bonds_matched['tmyld_Note']
bonds_matched['bidask_ratio_Spread'] = bonds_matched['bidask_ratio_Bond'] - bonds_matched['bidask_ratio_Note']
bonds_matched['tcouprt_Spread'] = bonds_matched['tcouprt_Bond'] - bonds_matched['tcouprt_Note']
bonds_matched['d2mat_Spread'] = bonds_matched['tmatdt_Bond']-bonds_matched['tmatdt_Note']
bonds_matched['d2mat_Spread'] = bonds_matched['d2mat_Spread'] // np.timedelta64(1, "D")
bonds_matched['year'] = bonds_matched['mcaldt'].dt.year

##### REGRESSION ANALYSIS (BONDS-NOTES) #####

Y = bonds_matched['tmyld_Spread']
X = bonds_matched[['bidask_ratio_Spread','tcouprt_Spread','y2mat_Bond','T10Y2YM_Note']]
# X = bonds_matched[['bidask_ratio_Spread','tcouprt_Spread','T10Y2YM_Note']]

X = sm.add_constant(X,prepend=True)

model = sm.OLS(Y,X,)
results = model.fit(cov_type='HC1')

print('\n\n ****** BONDS - NOTES REGRESSION ****** \n\n')
print(results.summary())

# print('\n\n ****** LaTeX VERSION ****** \n\n')
# print(results.summary().as_latex())

##### CALCULATING RELATIVE RATES (BONDS-BILLS) #####

bb_matched['tmyld_Spread'] = bb_matched['tmyld_Bond'] - bb_matched['tmyld_Bill']
bb_matched['bidask_ratio_Spread'] = bb_matched['bidask_ratio_Bond'] - bb_matched['bidask_ratio_Bill']
bb_matched['tcouprt_Spread'] = bb_matched['tcouprt_Bond'] - bb_matched['tcouprt_Bill']
bb_matched['d2mat_Spread'] = bb_matched['tmatdt_Bond']-bb_matched['tmatdt_Bill']
bb_matched['d2mat_Spread'] = bb_matched['d2mat_Spread'] // np.timedelta64(1, "D")
bb_matched['year'] = bb_matched['mcaldt'].dt.year

##### REGRESSION ANALYSIS (BONDS-NOTES) #####

Y = bb_matched['tmyld_Spread']
X = bb_matched[['bidask_ratio_Spread','tcouprt_Spread','m2mat_Bond','T10Y2YM_Bond']]
# X = bonds_matched[['bidask_ratio_Spread','tcouprt_Spread','T10Y2YM_Note']]

X = sm.add_constant(X,prepend=True)

model = sm.OLS(Y,X,)
results = model.fit(cov_type='HC1')

print('\n\n ****** BONDS - BILLS REGRESSION ****** \n\n')
print(results.summary())

##### DATA VISUALIZATIONS ####
# -> Not particularly useful
# print('\n\n DATA VISUALIZATIONS \n\n')

# plt.scatter(bonds_matched['bidask_ratio_Spread'],bonds_matched['tmyld_Spread'])
# plt.xlabel('Relative Bid-Ask Ratio Spread')
# plt.ylabel('YTM Spread')
# plt.show()