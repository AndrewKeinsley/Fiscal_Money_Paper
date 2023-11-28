import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
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
start = datetime.datetime(2018,1,1)
end = datetime.datetime(2020,12,31)

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

# CONTRIBUTION ANALYSIS #####

## Contribution to the growth of the Fisher Quantity Index
### The sum is the total growth of the Fisher Quantity Index, not the percent contribution
### To calculate the percent contribution, divide each row by the sum of the row
CONTR = (Weight_Laspeyres.multiply(PI_1, axis=0) + Weight_Paashe.multiply(PI_2, axis=0))*QT_growth

CONTR_RATIO = CONTR.div(CONTR.sum(axis=1), axis=0)*100      # In percent

### Checking that the sum of the contributions equals 100
check_CONTR_SUM = CONTR_RATIO.sum(axis=1)

## Constructing a dataframe with the contributions based on type


CONTR_TYPE = pd.DataFrame({'Date':data_raw['Date'],'Bills': CONTR_RATIO.filter(like='Bill', axis=1).sum(axis=1)})

CONTR_TYPE['Notes'] = CONTR_RATIO.filter(regex='^Note[^i]', axis=1).sum(axis=1)
CONTR_TYPE['Bonds'] = CONTR_RATIO.filter(regex='^Bond[^i]', axis=1).sum(axis=1)
CONTR_TYPE['iNotes'] = CONTR_RATIO.filter(like='iNote', axis=1).sum(axis=1)
CONTR_TYPE['iBonds'] = CONTR_RATIO.filter(like='iBond', axis=1).sum(axis=1)

check_CONTRTYPE_SUM = CONTR_TYPE.sum(axis=1)
### Resetting the index to the date column
CONTR_TYPE = CONTR_TYPE.set_index('Date')

# PLOTS #####

## Plotting the stacked area chart
fig, ax = plt.subplots()
ax.stackplot(CONTR_TYPE.index, CONTR_TYPE['Bills'], CONTR_TYPE['Notes'], CONTR_TYPE['Bonds'], CONTR_TYPE['iNotes'], CONTR_TYPE['iBonds'], labels=['Bills', 'Notes', 'Bonds', 'iNotes', 'iBonds'])
ax.legend(loc='lower left')
plt.xlim((start, end)) 
plt.xticks(rotation=45)
plt.ylim((CONTR_TYPE.loc[start:end].min().min(), CONTR_TYPE.loc[start:end].max().max()))
CONTR_TYPE.loc[start:end]
plt.show()

