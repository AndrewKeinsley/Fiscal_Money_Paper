import pandas as pd

dataframe_raw = pd.read_excel('Treasury_Data.xlsx')


# Narrowing the dataframe to only Bonds, Notes, and Bills, respectively.
df_narrowed = dataframe_raw.loc[dataframe_raw['itype'].isin([1,2,4])]

# Reducing the number of variables for ease of use
df = df_narrowed[['crspid','tcusip','tmatdt','tcouprt','itype','tmbid','tmask','tmaccint','tmduratn','tmpcyld']]

# Dropping any observations with NaN values
df = df.dropna()

# Checking the data for anymore issues
df.describe()