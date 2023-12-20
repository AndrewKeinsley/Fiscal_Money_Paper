import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
import datetime
from datetime import timedelta
# import statsmodels.api as sm
import pandas_datareader.data as web
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import VAR
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


# IMPORTING DATA ##############################################################
# Here I import the data from the 'Analysis/Contribution_Analysis.py' file as 
# well as from FRED.

## DATA FROM 'Analysis/Contribution_Analysis.py' ##############################
# Data from 'Analysis/Contribution_Analysis.py' includes the quantities and 
# user costs (prices) for each of the asset classes in the Treasury dataset. 

Q = pd.read_pickle('QT_t1.pkl').set_index('Date')
P = pd.read_pickle('UC_t1.pkl').set_index('Date')

## DATA FROM FRED #############################################################
# Data from FRED includes the following variables, which are monthly unless 
# otherwise noted:
#   - AAA: Moody's Seasoned Aaa Corporate Bond Yield (Percent, NSA)
#   - GS10: Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, 
#           Quoted on an Investment Basis (Percent, NSA)
#   - BAA: Moody's Seasoned Baa Corporate Bond Yield (Percent, NSA)
#   - SPASTT01USM657N: Share Prices: All Shares/Broad: Total for United States 
#                      (MoM Growth Rate, NSA)
#   - MTSDS133FMS: Federal Surplus or Deficit [-] (Millions, NSA)
#   - PCE: Personal Consumption Expenditures (Billions, SA)
#   - WILL5000INDFC: Wilshire 5000 Total Market Full Cap Index (Daily, Index, NSA)
#   - DTWEXM: Nominal Major Currencies U.S. Dollar Index (Daily, Index, NSA)
#   - DTWEXAFEGS: Nominal Advanced Foreign Economies U.S. Dollar Index (Daily, Index, NSA)

### START AND END DATES #######################################################
start_date = Q.index[0]
end_date = Q.index[-1]

#### Monthly Data #############################################################
Z_monthly = web.DataReader(['AAA','GS10','BAA','SPASTT01USM657N','MTSDS133FMS','PCE'],'fred',start_date,end_date)
Z_monthly.dropna(inplace=True)

#### Daily Data ###############################################################
Z_daily = web.DataReader(['WILL5000INDFC','DTWEXM','DTWEXAFEGS'],'fred',start_date,end_date)

# DATA PREPARATION ############################################################
# There are a number of alterations that need to be made to the data before it
# can be used in the analysis. The final varialbes will represent:
#   - Liquidity: AAA-GS10
#   - Safety: BAA-AAA
#   - Stock Market Returns: SPASTT01USM657N
#   - Federal Deficit-to-PCE: MTSDS133FMS/PCE*100
#       - First the Deficit must be seasonally adjusted
#   - Stock Market Volatility Index: var(WILL5000INDFC)
#       - Rolling 360-day variance, then averaged over the month
#   - Dollar Index: DTWEXM and DTWEXAFEGS
#       - First the two indices must be combined into a single index
#       - This is done by re-indexing them to a common point in time, then
#         averaging the two indices where they overlap

## PREPARING THE EXOGENOUS DATA ###############################################
Z = pd.DataFrame(index=Z_monthly.index)
Z['Liquidity'] = Z_monthly['AAA'] - Z_monthly['GS10']
Z['Safety'] = Z_monthly['BAA'] - Z_monthly['AAA']
Z['Stock_Return'] = Z_monthly['SPASTT01USM657N']

decomp = seasonal_decompose(Z_monthly['MTSDS133FMS'],model='additive')
Z['Deficit_Ratio'] = (decomp.trend)/(Z_monthly['PCE']*1000)*100

Z_daily['WILL5000INDFC'].interpolate(method='linear',limit_direction='both',inplace=True)
Stock_Volatility = Z_daily['WILL5000INDFC'].rolling(window='360D').var().resample('M').mean()
Stock_Volatility.index = Stock_Volatility.index.to_period('M').to_timestamp()
Z['Stock_Volatility'] = np.log(Stock_Volatility).diff()

Z_daily['DTWEXM'] = Z_daily['DTWEXM']/Z_daily.loc['2010-01-04','DTWEXM']*100
Z_daily['DTWEXAFEGS'] = Z_daily['DTWEXAFEGS']/Z_daily.loc['2010-01-04','DTWEXAFEGS']*100
# Z_daily[['DTWEXM','DTWEXAFEGS']] = Z_daily[['DTWEXM','DTWEXAFEGS']]/Z_daily.loc['2010-01-01',['DTWEXM','DTWEXAFEGS']]*100
Dollar_Index = Z_daily[['DTWEXM','DTWEXAFEGS']].mean(axis=1).resample('M').mean()
Dollar_Index.index = Dollar_Index.index.to_period('M').to_timestamp()
Z['Dollar_Index'] = Dollar_Index.diff()#.pct_change()*100

# Move the index to the end of the month
Z.index = Z.index.to_period('M').to_timestamp('M') 

# Drop any missing values
Z = Z.dropna()

## PREPARING THE ENDOGENOUS DATA ##############################################
# The endogenous data is the quantity and user cost data from the Treasury
# dataset. The data is already in the correct format, but it needs to be aggregated
# into "sectors" as in Shapiro (2022). The categories will be (in quarters to maturity):
#   - Bills
#       - 1, 2-4, 5-8
#   - Notes
#       - 1-20, 21-40
#   - Bonds
#       - 1-40, 41-80, 81-120
#   - TIPS: Ignoring these for now

### AGGREGATING THE DATA ######################################################

# Defining the ranges for each 'sector'
categories = {
    'Bill_Short': range(1, 2),
    'Bill_Long': range(2, 5),
    'Note_Short': range(1, 21),
    'Note_Long': range(21, 41),
    'Bond_Short': range(1, 41),
    'Bond_Medium': range(41, 81),
    'Bond_Long': range(81, 121)
}

# Initialize the DataFrames
Q_Agg = pd.DataFrame(index=Q.index)
P_Agg = pd.DataFrame(index=P.index)

# Loop over each category
for category, r in categories.items():
    # Aggregate the quantity data
    Q_Agg[category] = Q[[f'{category.split("_")[0]}{i}' for i in r]].sum(axis=1)
    
    # Calculate the weighted average for the user cost data
    weights = Q[[f'{category.split("_")[0]}{i}' for i in r]].values
    values = P[[f'{category.split("_")[0]}{i}' for i in r]].values
    P_Agg[category] = np.average(values, weights=weights, axis=1)

# Take the natural logarithm of the quantity data
Q_Agg = np.log(Q_Agg)
    

# VAR MODEL ###################################################################
# The VAR model is estimated using the statsmodels package. The model is
# estimated using the following variables:
#   - Liquidity: AAA-GS10
#   - Safety: BAA-AAA
#   - Stock Market Returns: SPASTT01USM657N
#   - Federal Deficit-to-PCE: MTSDS133FMS/PCE*100
#   - Stock Market Volatility Index: var(WILL5000INDFC)
#   - Dollar Index: DTWEXM and DTWEXAFEGS
    
# Initialize a dictionary to store the results
results = {}
residuals = {}

# # Loop over each column
# for column in Q_Agg.columns:
#     # Combine the current column from Q_Agg and P_Agg
#     data = pd.concat([Q_Agg[[column]], P_Agg[[column]], Z.shift(1)], axis=1)
#     data = data.dropna()
#     # Create the VAR model
#     model = VAR(data)
    
#     # Fit the model with twelve lags
#     results[column] = model.fit(maxlags=1)
    
#     # Store the residuals
#     residuals[column] = results[column].resid

# Loop over each column
for column in Q_Agg.columns:
    # Create the lagged variables
    Q_lags = pd.concat([Q_Agg[[column]].shift(i) for i in range(1, 13)], axis=1)
    P_lags = pd.concat([P_Agg[[column]].shift(i) for i in range(1, 13)], axis=1)
    Z_lag = Z.shift(1)

    # Combine the lagged variables
    X = pd.concat([Q_lags, P_lags, Z_lag], axis=1).dropna()
    X = add_constant(X)  # Add a constant term

    # Create the dependent variable
    y = pd.concat([Q_Agg[[column]], P_Agg[[column]]], axis=1).loc[X.index]

    # Create and fit the OLS model
    model = OLS(y, X)
    results[column] = model.fit()

    # Store the residuals
    residuals[column] = results[column].resid