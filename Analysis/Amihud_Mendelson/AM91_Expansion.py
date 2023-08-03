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
df = df[posvals].reset_index()

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

