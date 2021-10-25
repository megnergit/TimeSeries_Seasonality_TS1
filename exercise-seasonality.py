#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# **This notebook is an exercise in the [Time Series](https://www.kaggle.com/learn/time-series) course.  You can reference the tutorial at [this link](https://www.kaggle.com/ryanholbrook/seasonality).**
#
# ---
#

# # Introduction #
#
# Run this cell to set everything up!

# In[ ]:


# In[1]:


# Setup feedback system
from scipy.signal import periodogram
import plotly.graph_objs as go
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from learntools.time_series.utils import plot_periodogram, seasonal_plot
from learntools.time_series.style import *  # plot style settings
from pathlib import Path
from learntools.time_series.ex3 import *
from learntools.core import binder
binder.bind(globals())

# Setup notebook


comp_dir = Path('../input/store-sales-time-series-forecasting')

holidays_events = pd.read_csv(
    comp_dir / "holidays_events.csv",
    dtype={
        'type': 'category',
        'locale': 'category',
        'locale_name': 'category',
        'description': 'category',
        'transferred': 'bool',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
holidays_events = holidays_events.set_index('date').to_period('D')

store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(
    ['store_nbr', 'family', 'date']).sort_index()
average_sales = (
    store_sales
    .groupby('date').mean()
    .squeeze()
    .loc['2017']
)


# In[2]:


DATA_DIR = Path('../input/store-sales-time-series-forecasting')
holidays_events = pd.read_csv(DATA_DIR/'holidays_events.csv',
                              dtype={'type': 'category',
                                     'locale': 'category',
                                     'locale_name': 'category',
                                     'description': 'category',
                                     'transferred': 'bool',
                                     },
                              parse_dates=['date'], index_col='date',
                              infer_datetime_format=True).to_period('D').sort_index()

store_sales = pd.read_csv(DATA_DIR/'train.csv', parse_dates=['date'],
                          dtype={'store_nbr': 'category',
                                   'family': 'category'
                                 },
                          index_col='date').to_period('D').sort_index()

store_sales.set_index(['store_nbr', 'family'], append=True, inplace=True)
# period has to be set before multiple index is set

# one can do like this. '2017'


average_sales = store_sales.groupby('date').mean().loc['2017', ['sales']]
average_sales = pd.DataFrame(average_sales)
print(type(average_sales))
average_sales


# In[3]:


# - parse
# - index
# - set period to 'D'
# - dtype (category to category)


# In[4]:


average_sales = (
    store_sales
    .groupby('date').mean()
    .squeeze()
    .loc['2017']
)


# In[5]:


average_sales.index.asfreq('W')


# In[6]:


trace = go.Scatter(x=average_sales.index.to_timestamp(),
                   y=average_sales['sales'])
data = [trace]
fig = go.Figure(data=data)
fig.show()


# In[7]:


X = average_sales.copy()
X_week = pd.Index(X.asfreq('W').index, name='week')
X['day_of_week'] = X.index.dayofweek
X.set_index(X_week, append=True, inplace=True)


# In[8]:


def color_dict(date_index):
    '''
    date_index : index of pd.DataFrame
    '''
    a = pd.Series(range(len(date_index)))
    a_min = a.min()
    a_max = a.max()

    c_a = (0.8 / (a_max - a_min) * (a - a_min) + 0.1) * 256 ** 3
    c_a = c_a.astype(int)

    c_a = ['#'+hex(c).replace('0x', '') for c in c_a]

    d = {}
    dump = [d.update({x: c}) for x, c in zip(a, c_a)]

    return d


# In[9]:

colors = color_dict(X.index.get_level_values(1))


data = []
for i, s in enumerate(X.index.get_level_values(1)):
    x = X.iloc[X.index.get_level_values(1) == s]['day_of_week']
    y = X.iloc[X.index.get_level_values(1) == s]['sales']
    color = colors[i]

    trace = go.Scatter(x=x, y=y, line=dict(color=color))
    data.append(trace)

fig = go.Figure(data=data)
fig.show()


# In[10]:


average_sales = store_sales.groupby('date').mean().loc['2017', ['sales']]


# In[11]:


print(pd.Timedelta('1Y'))
print(pd.Timedelta('1D'))
fs = pd.Timedelta('1Y') / pd.Timedelta('1D')
print(fs)

frequencies, spectrum = periodogram(average_sales['sales'],
                                    fs=fs,
                                    detrend='linear',
                                    window='boxcar',
                                    scaling='spectrum',)


# In[12]:


print(pd.Timedelta('1Y'))
print(pd.Timedelta('1D'))
fs = pd.Timedelta('1Y') / pd.Timedelta('1D')
print(fs)

frequencies, spectrum = periodogram(average_sales['sales'],
                                    fs=fs,
                                    detrend='linear',
                                    window='boxcar',
                                    scaling='spectrum',)


trace = go.Scatter(x=frequencies, y=spectrum, line_shape='hvh')
data = [trace]
layout = go.Layout(xaxis=dict(type='log'))
fig = go.Figure(data=data, layout=layout)
fig.show()


# In[ ]:


# -------------------------------------------------------------------------------
#
# Examine the following seasonal plot:

# In[13]:


X = average_sales.to_frame()
X["week"] = X.index.week
X["day"] = X.index.dayofweek
seasonal_plot(X, y='sales', period='week', freq='day')


# And also the periodogram:

# In[ ]:


plot_periodogram(average_sales)


# # 1) Determine seasonality
#
# What kind of seasonality do you see evidence of? Once you've thought about it, run the next cell for some discussion.

# In[ ]:


# View the solution (Run this cell to receive credit!)
q_1.check()


# In[ ]:


# In[ ]:


fourier = CalendarFourier(freq='A', order=4)

dp = DeterministicProcess(
    index=average_sales.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True,
)


# -------------------------------------------------------------------------------
#
# # 2) Create seasonal features
#
# Use `DeterministicProcess` and `CalendarFourier` to create:
# - indicators for weekly seasons and
# - Fourier features of order 4 for monthly seasons.

# In[ ]:


# In[ ]:


X = average_sales.copy()
X_week = pd.Index(X.asfreq('W').index, name='week')
X['day_of_week'] = X.index.dayofweek
X.set_index(X_week, append=True, inplace=True)


y = average_sales['sales'].copy()

# YOUR CODE HERE
fourier = CalendarFourier(freq='M', order=4)
dp = DeterministicProcess(
    index=y.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True,
)
X = dp.in_sample()

# Check your answer
q_2.check()


# Now run this cell to fit the seasonal model.

# In[ ]:


X.head(3)


# In[ ]:


# In[ ]:


model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X).squeeze()


# In[ ]:


# y_pred


# In[ ]:


trace_0 = go.Scatter(x=average_sales.index.to_timestamp(),
                     y=average_sales['sales'])
trace_1 = go.Scatter(x=X.index.get_level_values(0).to_timestamp(), y=y_pred)
data = [trace_0, trace_1]
layout = go.Layout(height=640, width=1440)
fig = go.Figure(data=data, layout=layout)
fig.show()


# In[ ]:


X.index.get_level_values(0).to_timestamp()


# In[ ]:


model = LinearRegression().fit(X, y)
y_pred = pd.Series(
    model.predict(X),
    index=X.index,
    name='Fitted',
)

y_pred = pd.Series(model.predict(X), index=X.index)
ax = y.plot(**plot_params, alpha=0.5,
            title="Average Sales", ylabel="items sold")
ax = y_pred.plot(ax=ax, label="Seasonal")
ax.legend()


# -------------------------------------------------------------------------------
#

# Removing from a series its trend or seasons is called **detrending** or **deseasonalizing** the series.
#
# Look at the periodogram of the deseasonalized series.

# In[ ]:


y_deseason = y - y_pred

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))
ax1 = plot_periodogram(y, ax=ax1)
ax1.set_title("Product Sales Frequency Components")
ax2 = plot_periodogram(y_deseason, ax=ax2)
ax2.set_title("Deseasonalized")


# In[ ]:


y_deseason = y.values.squeeze() - y_pred.squeeze()
# y_deseason


# In[ ]:


trace_0 = go.Scatter(x=average_sales.index.to_timestamp(),
                     y=average_sales['sales'])
trace_1 = go.Scatter(x=X.index.get_level_values(0).to_timestamp(), y=y_pred)
trace_3 = go.Scatter(x=X.index.get_level_values(
    0).to_timestamp(), y=y_deseason)

data = [trace_0, trace_1, trace_3]
layout = go.Layout(height=640, width=1440)
fig = go.Figure(data=data, layout=layout)
fig.show()


# In[ ]:


print(pd.Timedelta('1Y'))
print(pd.Timedelta('1D'))
fs = pd.Timedelta('1Y') / pd.Timedelta('1D')
print(fs)

frequencies, spectrum = periodogram(y_deseason,
                                    fs=fs,
                                    detrend='linear',
                                    window='boxcar',
                                    scaling='spectrum',)


trace = go.Scatter(x=frequencies, y=spectrum, line_shape='hvh')
data = [trace]
layout = go.Layout(xaxis=dict(type='log'))
fig = go.Figure(data=data, layout=layout)
fig.show()


# # 3) Check for remaining seasonality
#
# Based on these periodograms, how effectively does it appear your model captured the seasonality in *Average Sales*? Does the periodogram agree with the time plot of the deseasonalized series?

# In[ ]:


# View the solution (Run this cell to receive credit!)
q_3.check()


# -------------------------------------------------------------------------------
#
# The *Store Sales* dataset includes a table of Ecuadorian holidays.

# In[ ]:


holidays_events.head(3)


# In[ ]:


len(holidays_events)
print(holidays_events['locale'].unique())
print(holidays_events['type'].unique())
holidays_events['locale_name'].unique()
holidays_events['description'].unique()


# In[ ]:


holidays_in_ecuador = holidays_events[holidays_events['locale'].isin(
    ['National', 'Regional'])]
# holidays_in_ecuador = holidays_events[holidays_events['locale'].isin(['National'])]

holidays = holidays_in_ecuador[(holidays_in_ecuador.index >= y.index.min()) &
                               (holidays_in_ecuador.index <= y.index.max())]


# In[ ]:


yp


# In[ ]:


yp = pd.DataFrame(y.squeeze()-y_pred, index=y.index)


# In[ ]:


# National and regional holidays in the training set
holidays = (
    holidays_events
    .query("locale in ['National', 'Regional']")
    .loc['2017':'2017-08-15', ['description']]
    .assign(description=lambda x: x.description.cat.remove_unused_categories())
)

display(holidays)


# In[ ]:


yp.head(3)


# From a plot of the deseasonalized *Average Sales*, it appears these holidays could have some predictive power.

# In[ ]:


trace_0 = go.Scatter(x=average_sales.index.to_timestamp(),
                     y=average_sales['sales'])
trace_1 = go.Scatter(x=X.index.get_level_values(0).to_timestamp(), y=y_pred)
trace_3 = go.Scatter(x=X.index.get_level_values(
    0).to_timestamp(), y=yp['sales'])
trace_4 = go.Scatter(x=holidays.index.to_timestamp(),
                     y=yp.loc[holidays.index, 'sales'], mode='markers')

data = [trace_0, trace_1, trace_3, trace_4]
# data = [trace_3, trace_4]

layout = go.Layout(height=640, width=1440)
fig = go.Figure(data=data, layout=layout)
fig.show()


# In[ ]:


ax = y_deseason.plot(**plot_params)
plt.plot_date(holidays.index, y_deseason[holidays.index], color='C3')
ax.set_title('National and Regional Holidays')


# # 4) Create holiday features
#
# What kind of features could you create to help your model make use of this information? Code your answer in the next cell. (Scikit-learn and Pandas both have utilities that should make this easy. See the `hint` if you'd like more details.)
#

# In[ ]:


X = average_sales.copy()
X_week = pd.Index(X.asfreq('W').index, name='week')
X['day_of_week'] = X.index.dayofweek
X.set_index(X_week, append=True, inplace=True)


y = average_sales['sales'].copy()

# YOUR CODE HERE
fourier = CalendarFourier(freq='M', order=4)
dp = DeterministicProcess(
    index=y.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True,
)
X = dp.in_sample()


# In[ ]:


X


# In[ ]:


X_holidays = pd.get_dummies(holidays)
X2 = X.join(X_holidays, how='left').fillna(0.0)

# Check your answer
q_4.check()


# In[ ]:


# Lines below will give you a hint or solution code
q_4.hint()
q_4.hint(2)
q_4.solution()


# Use this cell to fit the seasonal model with holiday features added. Do the fitted values seem to have improved?

# In[ ]:


model = LinearRegression()
model.fit(X2, y)
y_pred = model.predict(X2)


# In[ ]:


yp2 = y.values - y_pred


# In[ ]:


trace_0 = go.Scatter(x=average_sales.index.to_timestamp(),
                     y=average_sales['sales'])
trace_1 = go.Scatter(x=X2.index.get_level_values(0).to_timestamp(), y=y_pred)
trace_3 = go.Scatter(x=X2.index.get_level_values(0).to_timestamp(), y=yp2)
trace_4 = go.Scatter(x=holidays.index.to_timestamp(),
                     y=yp.loc[holidays.index, 'sales'], mode='markers')

data = [trace_0, trace_1, trace_3, trace_4]
# data = [trace_3, trace_4]
# data = [trace_0, trace_1, trace_3]

layout = go.Layout(height=640, width=1440)
fig = go.Figure(data=data, layout=layout)
fig.show()


# In[ ]:


model = LinearRegression().fit(X2, y)
y_pred = pd.Series(
    model.predict(X2),
    index=X2.index,
    name='Fitted',
)

y_pred = pd.Series(model.predict(X2), index=X2.index)
ax = y.plot(**plot_params, alpha=0.5,
            title="Average Sales", ylabel="items sold")
ax = y_pred.plot(ax=ax, label="Seasonal")
ax.legend()


# -------------------------------------------------------------------------------
#
# # (Optional) Submit to Store Sales competition
#
# This part of the exercise will walk you through your first submission to this course's companion competition: [**Store Sales - Time Series Forecasting**](https://www.kaggle.com/c/29781). Submitting to the competition isn't required to complete the course, but it's a great way to try out your new skills.
#
# The next cell creates a seasonal model of the kind you've learned about in this lesson for the full *Store Sales* dataset with all 1800 time series.

# In[ ]:


y = store_sales.unstack(['store_nbr', 'family']).loc["2017"]

# Create training data
fourier = CalendarFourier(freq='M', order=4)
dp = DeterministicProcess(
    index=y.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True,
)
X = dp.in_sample()
X['NewYear'] = (X.index.dayofyear == 1)

model = LinearRegression(fit_intercept=False)
model.fit(X, y)
y_pred = pd.DataFrame(model.predict(X), index=X.index, columns=y.columns)


# You can use this cell to see some of its predictions.
#

# In[ ]:


STORE_NBR = '1'  # 1 - 54
FAMILY = 'PRODUCE'
# Uncomment to see a list of product families
# display(store_sales.index.get_level_values('family').unique())

ax = y.loc(axis=1)['sales', STORE_NBR, FAMILY].plot(**plot_params)
ax = y_pred.loc(axis=1)['sales', STORE_NBR, FAMILY].plot(ax=ax)
ax.set_title(f'{FAMILY} Sales at Store {STORE_NBR}')


# Finally, this cell loads the test data, creates a feature set for the forecast period, and then creates the submission file `submission.csv`.

# In[ ]:


df_test = pd.read_csv(
    comp_dir / 'test.csv',
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'onpromotion': 'uint32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
df_test['date'] = df_test.date.dt.to_period('D')
df_test = df_test.set_index(['store_nbr', 'family', 'date']).sort_index()

# Create features for test set
X_test = dp.out_of_sample(steps=16)
X_test.index.name = 'date'
X_test['NewYear'] = (X_test.index.dayofyear == 1)


y_submit = pd.DataFrame(model.predict(
    X_test), index=X_test.index, columns=y.columns)
y_submit = y_submit.stack(['store_nbr', 'family'])
y_submit = y_submit.join(df_test.id).reindex(columns=['id', 'sales'])
y_submit.to_csv('submission.csv', index=False)


# To test your forecasts, you'll need to join the competition (if you haven't already). So open a new window by clicking on [this link](https://www.kaggle.com/c/29781). Then click on the **Join Competition** button.
#
# Next, follow the instructions below:
# 1. Begin by clicking on the **Save Version** button in the top right corner of the window.  This will generate a pop-up window.
# 2. Ensure that the **Save and Run All** option is selected, and then click on the **Save** button.
# 3. This generates a window in the bottom left corner of the notebook.  After it has finished running, click on the number to the right of the **Save Version** button.  This pulls up a list of versions on the right of the screen.  Click on the ellipsis **(...)** to the right of the most recent version, and select **Open in Viewer**.  This brings you into view mode of the same page. You will need to scroll down to get back to these instructions.
# 4. Click on the **Output** tab on the right of the screen.  Then, click on the file you would like to submit, and click on the **Submit** button to submit your results to the leaderboard.
#
# You have now successfully submitted to the competition!
#
# If you want to keep working to improve your performance, select the **Edit** button in the top right of the screen. Then you can change your code and repeat the process. There's a lot of room to improve, and you will climb up the leaderboard as you work.
#

# # Keep Going #
#
# [**Use time series as features**](https://www.kaggle.com/ryanholbrook/time-series-as-features) to capture cycles and other kinds of serial dependence.
