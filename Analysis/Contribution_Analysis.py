import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import datetime
from datetime import timedelta
# import statsmodels.api as sm
import pandas_datareader.data as web

# IMPORTING DATA #####

## Treasury Data (t = 1)
UC_t1 = pd.read_excel('UserCosts.xlsx')
QT_t1 = pd.read_excel('Quantities.xlsx')

data_raw = QT_t1

### Dropping the Date Column
UC_t1 = UC_t1.drop('Date', axis=1)
QT_t1 = QT_t1.drop('Date', axis=1)

### Adding 1 to all values in QT_t1 to avoid division by zero
QT_t1 = QT_t1+1

## Creating lagged Treasury Data (t = 0)
UC_t0 = UC_t1.shift(1)
QT_t0 = QT_t1.shift(1)

## Additional Time Series Data
start = datetime.datetime(1984,1,1)
end = datetime.datetime(1986,12,31)
## Recession data from FRED 
rec_data = (web.DataReader(['USREC'], 'fred', start, end))

# PRICE INDICES #####

## Paashe Price Index
Paashe_num = (QT_t1*UC_t1).sum(axis=1)  # Numerator
Paashe_den = (QT_t1*UC_t0).sum(axis=1)  # Denominator
Paashe = Paashe_num/Paashe_den

## Laspeyres Price Index
Laspeyres_num = (QT_t0*UC_t1).sum(axis=1)  # Numerator
Laspeyres_den = (QT_t0*UC_t0).sum(axis=1)  # Denominator
Laspeyres = Laspeyres_num/Laspeyres_den

## Fisher Price Index
Fisher = np.sqrt(Paashe*Laspeyres)

# WEIGHTS #####

## Weighting Matrices
Weight_Laspeyres = (QT_t0*UC_t0).div(Laspeyres_den, axis=0)
Weight_Paashe = (QT_t0*UC_t1).div(Laspeyres_num, axis=0)

### Checking that the sum of the weights is 1
check_WL_SUM = Weight_Laspeyres.sum(axis=1)
check_WP_SUM = Weight_Paashe.sum(axis=1)

## PRICE INDEX RATIOS
PI_1 = Fisher/(Laspeyres+Paashe)
PI_2 = Laspeyres/(Laspeyres+Fisher)

# QUANTITY GROWTH RATES #####
QT_growth = QT_t1.div(QT_t0, axis=0)-1

# CONTRIBUTION ANALYSIS: FISHER INDEX #####

## Contribution to the growth of the Fisher Quantity Index
### The sum is the total growth of the Fisher Quantity Index, not the percent contribution
### To calculate the percent contribution, divide each row by the sum of the row
CONTR = (Weight_Laspeyres.multiply(PI_1, axis=0) + Weight_Paashe.multiply(PI_2, axis=0))*QT_growth

# CONTR_RATIO = CONTR.div(CONTR.sum(axis=1), axis=0)*100      # In percent of total
CONTR_RATIO = CONTR*100      # In levels of growth

### Checking that the sum of the contributions equals the total
check_CONTR_SUM = CONTR_RATIO.sum(axis=1)

## Constructing a dataframe with the contributions based on type of security
CONTR_TYPE = pd.DataFrame({'Date':data_raw['Date'],'Bills': CONTR_RATIO.filter(like='Bill', axis=1).sum(axis=1)})

CONTR_TYPE['Notes'], CONTR_TYPE['Bonds'], CONTR_TYPE['iNotes'], CONTR_TYPE['iBonds'] = [CONTR_RATIO.filter(regex=regex, axis=1).sum(axis=1) for regex in ['^Note[^i]', '^Bond[^i]', 'iNote', 'iBond']]

check_CONTRTYPE_SUM = CONTR_TYPE.sum(axis=1)
### Resetting the index to the date column
CONTR_TYPE = CONTR_TYPE.set_index('Date')


# CONTRIBUTION ANALYSIS: SIMPLE SUM ###

## Contribution to the growth of the simple sum quantity index
DIFF_SS = QT_t1 - QT_t0
CONTR_SS = DIFF_SS.div(QT_t0.sum(axis=1),axis=0)*100

## Constructing a dataframe with the contributions based on type of security
CONTR_SS_TYPE = pd.DataFrame({'Date':data_raw['Date'],'Bills': CONTR_SS.filter(like='Bill', axis=1).sum(axis=1)})

CONTR_SS_TYPE['Notes'], CONTR_SS_TYPE['Bonds'], CONTR_SS_TYPE['iNotes'], CONTR_SS_TYPE['iBonds'] = [CONTR_SS.filter(regex=regex, axis=1).sum(axis=1) for regex in ['^Note[^i]', '^Bond[^i]', 'iNote', 'iBond']]

check_CONTRTYPE_SS = CONTR_SS_TYPE.sum(axis=1)
## Resetting the index to the date column
CONTR_SS_TYPE = CONTR_SS_TYPE.set_index('Date')

# ISOLATING THE CONTRIBUTION TO MONETARY SERVICES #####
CONTR_MS_TYPE = CONTR_TYPE-CONTR_SS_TYPE

# PLOTS #####

## Plotting the individual time series

## Contribution to the growth of the Monetary Services Quantity Index
# Separate positive and negative values
df_positive = CONTR_MS_TYPE.clip(lower=0)
df_negative = CONTR_MS_TYPE.clip(upper=0)

# Initialize cumulative sum for positive and negative parts
cumulative_positive = np.zeros(len(CONTR_MS_TYPE))
cumulative_negative = np.zeros(len(CONTR_MS_TYPE))

# Define colors for each series
colors = ['blue', 'orange', 'green', 'red', 'purple']

# Determine bar width in terms of days
days_width = 20
width_in_days = timedelta(days=days_width)

# Plot
fig, ax = plt.subplots()

## Plotting the recession dates
# Determine recession periods
recs_start = rec_data[rec_data.USREC.diff() == 1].index
recs_end = rec_data[rec_data.USREC.diff() == -1].index

# If a recession is ongoing at the end of the series, add the last date to recs_end
if len(recs_start) > len(recs_end):
    recs_end = recs_end.append(pd.Index([rec_data.index[-1]]))

# Plotting the recession dates
for rec_start, rec_end in zip(recs_start, recs_end):
    ax.axvspan(rec_start, rec_end, color='0.85', alpha=0.5)

## Looping through the columns and ploting them as stacked bars
for i, column in enumerate(CONTR_MS_TYPE.columns):
    color = colors[i]
    ax.bar(CONTR_MS_TYPE.index, df_positive[column], bottom=cumulative_positive,  width=width_in_days, color=color, label=column)
    ax.bar(CONTR_MS_TYPE.index, df_negative[column], bottom=cumulative_negative,  width=width_in_days, color=color)
    
    # Update the cumulative sums
    cumulative_positive += df_positive[column].fillna(0)
    cumulative_negative += df_negative[column].fillna(0)
## Add a horizontal line at y=0
ax.axhline(0, color='black', linewidth=0.5)  
## Add a line for the total: Not great for longer sample periods
ax.plot(CONTR_MS_TYPE.index, CONTR_MS_TYPE.sum(axis=1), color='black', linewidth=1, label='Total')
## Formatting
ax.set_ylabel('Percentage Points')
plt.xlim((start, end))
### Automatically adjusting the y-axis limits
plt.ylim((cumulative_negative.loc[start:end].min()-0.2, cumulative_positive.loc[start:end].max()+0.2)) 
### Setting the x-axis ticks to include the minor ticks
ax.set_xticks(CONTR_MS_TYPE.loc[start:end].index, minor=True)
plt.xticks(rotation=45)
plt.legend(loc=0, frameon=False, ncol=2)
filename = 'Contribution_MS_'+str(start.year)+'-'+str(end.year)
plt.savefig(filename+'.jpg', format='jpg', dpi=300, bbox_inches='tight')
plt.show()