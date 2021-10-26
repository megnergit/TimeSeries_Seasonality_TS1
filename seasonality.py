# |------------------------------------------------------------------
# | # Tunnel Traffic  - Time Series Analysis TS1
# |------------------------------------------------------------------
# |
# | ## 1. Introduction
# |
# | This is a notebook to practice the routine procedures
# | commonly used in the time sequence analysis.

# | Temporal sequence consists of several components.
# | - Trend (gradual decrease / increase)
# | - Seasonality (hour of day, day of week, week of month, month of year, etc.)
# | - Cycles (up and down but with a specific time scale)
# | - Peculiarity (national holidays, etc.)
# |

# | The list above is the whole features that one can
# | predict by time-sequence analysis and machine-learning models.
# | A model can predict what only repeats, i.e., what happened before.
# | A model cannot learn from what did not happen yet.
# |
# | To deal with each elements of time sequence, we have
# |
# | - For __trend__ : Analytical fitting of the baselines (linear, polynomial, etc)
# | - For __seasonality__ : Fourier decomposition.
# | - For __cycle__ : Lags.
# | - For __peculiarity__ : Categorical features.
# |

# | In this notebook we will familiarize ourselves with
# |
# |  * Manipulation of index.
# |  * 'DeterministicProcess` in 'statsmodels' package.
# |    'DeterministicProcess` will be used to create 'time dummy'.
# |

# | 'Time dummy's are the indices of target parameters.
# | Index in other types of data is just a convenient unique address
# | of a record to identify it. Since indices do not contain
# | any useful information, they are usually dropped during the machine learning.
# | On the contrary, index has special meaning in the time sequence analysis.
# | It is the feature needed for the prediction.

# | The manipulation of index in time sequence analysis include
# | 
# | * Date parsing
# | * Set date as index
# | * Set period of date to 'D' (or 'M', 'W )
# |   + [offset aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)
# | * Set `dtype` (category features to `category`)
# |

# | The only features (except categorical ones) used to model the  temporal
# | behavior of the target parameter is the time, but the time in different
# | intervals.
# | 'DeterministicProcess` help us to quickly create `t` in `y=f(t)`.
# | It is called 'deterministic', because it is a feature that are prefixed.
# | One can use such features at the time of prediction, i.e.
# | if we would like to predict a sales on a Sunday, we can use the fact
# | that day is a Sunday. On the contrary, we cannot use a sales one day before,
# | on Saturday, if it is not published yet at the time the prediction.
# | The former (=being a Sunday) is a deterministic feature, while the sales on
# | Saturday is a non-deterministic feature.


# | ## 2. Task
# |
# |  We have a record of traffic of a tunnel. Model the temporal sequence of
# |  the traffic from the time features only.

# | ## 3. Data
# |
# | 1. A traffic of vehicles traveling through the Baregg Tunnel
# |    in Switzerland from November 2003 to November 2005.
# |

# | ## 4. Notebook
# -------------------------------------------------------
# | Import packages.

from pathlib import Path
import os
import pandas as pd
import numpy as np

from scipy.signal import periodogram
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
import statsmodels.api as sm

from IPython.display import display
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import kaleido
from kaggle_tsa.ktsa import *

# -------------------------------------------------------
# | Set up directories.

CWD = Path('/Users/meg/git7/trend/')
DATA_DIR = Path('../input/ts-course-data/')
KAGGLE_DIR = Path('ryanholbrook/ts-course-data/')
IMAGE_DIR = Path('./images')
HTML_DIR = Path('./html')

os.chdir(CWD)
set_cwd(CWD)

# -------------------------------------------------------
# | If the data is not downloaded yet, do so now.

set_data_dir(KAGGLE_DIR, CWD)
show_whole_dataframe(True)

# -------------------------------------------------------
# | Read the data, first as it is.

tunnel = pd.read_csv(DATA_DIR/'tunnel.csv')

# | Check the contents.

print(tunnel.info())
display(tunnel.head(3))

# | Index manipulation, parsing. Currently `Day` column is taken as
# | `object` type (=string). We will convert it to `datetime` type.
# | In the same time `Day` column is set to be the index.
# | The frequency of the index is now set to 'D' (=calendar day frequency).
# | By setting the frequency, when we use `.shift(1)` to the dataframe,
# | the dataframe will be shifted by that amount in the unit of the specified
# | frequency. The `dtype`  of `NumVehicles` is set to 'float32'.
# |

tunnel = pd.read_csv(DATA_DIR/'tunnel.csv',
                     dtype={'NumVehicles': 'float32'},
                     parse_dates=['Day'],
                     index_col='Day').to_period('D')

print(tunnel.info())
print(tunnel.index)
display(tunnel.head(3))

# -------------------------------------------------------
# | Let us have a quick look at the data.

trace_1 = go.Scatter(x=tunnel.index.to_timestamp(),
                     y=tunnel['NumVehicles'],
                     name='Tunnel traffic')

data = [trace_1]
layout = go.Layout(height=512)
fig = go.Figure(data=data, layout=layout)

fig.show()
fig.write_image(IMAGE_DIR/'fig1.png')

# -------------------------------------------------------
# | One can see
# |
# | - Weekly trend (common in-week movement).
# | - In-year trend. Decrease of traffic during
# |    + Christmas holidays
# |    + Easter holidays
# |    + Summer vacations
# |
# | Here we concentrate on reproducing in-week trend.
# | First let us have a look.

tunnel = pd.read_csv(DATA_DIR/'tunnel.csv',
                     dtype={'NumVehicles': 'float32'},
                     parse_dates=['Day'],
                     index_col='Day').to_period('D')

tunnel['dayofweek'] = tunnel.index.dayofweek
tunnel['Week'] = tunnel.index.asfreq('W')

data = []
day_name = ('Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So')

for i, w in tunnel.groupby('Week'):
    trace = go.Scatter(x=w['dayofweek'],
                       y=w['NumVehicles'],
                       name=i)

    data.append(trace)

layout = go.Layout(height=1024,
                   xaxis=dict(categoryarray=day_name))
fig = go.Figure(data=data, layout=layout)
fig.show()
fig.write_image(IMAGE_DIR/'fig2.png')

# -------------------------------------------------------
# | The trend is nicely seen in which
# | the traffic gradually goes up from Monday to Friday,
# | and decreases toward Saturday and Sunday.
# -------------------------------------------------------

# | We will make two more plots.
# | 1. periodogram
# | 2. lagplot (although we do not use lag feature here)

# | [To-do : function to check if the index is continuous and no missing dates.]
# | 

annual_freq = (
    'Annual (1)',
    'Semiannual (2)',
    'Quarterly (4)',
    'Bimonthly (6)',
    'Monthly (12)',
    'Biweekly (26)',
    'Weekly (52)',
    'Semiweekly (104)')

fs = pd.Timedelta('1Y')/pd.Timedelta('1D')
frequencies, spectrum = periodogram(tunnel['NumVehicles'],
                                    fs=fs,
                                    detrend='linear',
                                    window='boxcar',
                                    scaling='spectrum'
                                    )

# |__`periodogram`__:\
# | `detrend` {'linear', 'constant'}\
# | `window`  {'boxcar', 'gaussian', ...} `scipy.signal.get_window`[shape of window function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window)\
# | `scaling` {'density', 'spectrum'}\  
# | The units are [V<sup>2</sup>/Hz] for 'density' (power spectrum)\
# | and [V<sup>2</sup>] for 'spectrum'.

trace = go.Scatter(x=frequencies,
                   y=spectrum,
                   fill='tozeroy',
                   line=dict(color='coral'),
                   line_shape='hvh')

data = [trace]
layout = go.Layout(height=640,
                   font=dict(size=20),
                   xaxis=dict(type='log',
                              ticktext=annual_freq,
                              tickvals=[1, 2, 4, 6, 12, 26, 52, 104]))

fig = go.Figure(data=data, layout=layout)
fig.show()
fig.write_image(IMAGE_DIR/'fig3.png')
# -------------------------------------------------------
# | We can see
# | * strong weekly repetitions, and
# | * annual, semiannual, and quarterly trend.
# |

# -------------------------------------------------------
# | Next, a lag plot.

n_lag = 12
n_cols = 3
n_rows = n_lag // n_cols
sm.OLS.df_degree = 1
fig = make_subplots(cols=n_cols,
                    rows=n_rows,
                    vertical_spacing=0.04,
                    subplot_titles=[f'Lag {i}' for i in range(1, n_lag+1)]
                    )

xax = ['x' + str(i) for i in range(1, n_lag+1)]
xax[0] = 'x'

yax = ['y' + str(i) for i in range(1, n_lag+1)]
yax[0] = 'y'

trace = [go.Scatter(x=tunnel['NumVehicles'],
                    y=tunnel['NumVehicles'].shift(i),
                    marker=dict(opacity=0.7),
                    mode='markers',
                    xaxis=xax[i-1],
                    yaxis=yax[i-1]) for i in range(1, n_lag+1)]

trace_reg = [go.Scatter(x=sorted(tunnel['NumVehicles']),
                        y=sorted(sm.OLS(tunnel['NumVehicles'].shift(i).values,
                                        tunnel['NumVehicles'],
                                        missing='drop').fit().fittedvalues),
                        mode='lines',
                        xaxis=xax[i-1], yaxis=yax[i-1]) for i in range(1, n_lag+1)]

data = trace + trace_reg
layout = go.Layout(height=1024 * 2,
                   font=dict(size=20),
                   showlegend=False)

layout = fig.layout.update(layout)
fig = go.Figure(data=data, layout=layout)
fig.show()
fig.write_image(IMAGE_DIR/'fig4.png')

corr = [tunnel['NumVehicles'].autocorr(lag=i) for i in range(1, n_lag+1)]

_ = [print(f'Lag {i:-2}: {tunnel["NumVehicles"].autocorr(lag=i):5.3f}')
     for i in range(1, n_lag+1)]

# -------------------------------------------------------
# | There are 3 components seen in `Lag_1` plot.
# |
# | - a bit increased from the day before
# | - a bit degreased from the day before
# | - about the same from the day before
# |

# | The correlation is high in 7-days lag than 1-day lag.
# | It corresponds to the weekly repetitions of the traffic we saw before.
# |

# -------------------------------------------------------
# | Partial autocorrelation function.
# | This will be used to decide how many lags should be
# | included in the features. We do not use the information here.

# The example of a plot for partial autocorrelation function
# from native `statsmodels`.
# fig = plot_pacf(tunnel['NumVehicles'], lags=12)


x_pacf, x_conf = pacf(tunnel['NumVehicles'], nlags=12, alpha=0.05)
x_error = (x_conf - x_pacf.repeat(2).reshape(x_conf.shape))[:, 1]
x_sig = 1.96 / np.sqrt(len(tunnel['NumVehicles']))

lags_name = [f'Lag {i-1}' for i in range(1, n_lag+1)]

trace_1 = go.Bar(y=lags_name,
                 error_x=dict(type='data', array=x_error),
                 x=x_pacf,
                 marker_color='indianred',
                 opacity=0.8,
                 width=0.8,
                 orientation='h')

trace_2a = go.Scatter(y=[lags_name[0],
                         lags_name[-1],
                         lags_name[-1],
                         lags_name[0],
                         lags_name[0]],
                      x=[-x_sig, -x_sig, x_sig, x_sig, -x_sig],
                      line=dict(color='teal'),
                      fill='toself',
                      mode='lines')

layout = go.Layout(height=512, width=800,
                   font=dict(size=20),
                   showlegend=False,
                   xaxis=dict(range=[-1.1, 1.1]))

data = [trace_1, trace_2a]

fig = go.Figure(data=data, layout=layout)
fig.show()
fig.write_image(IMAGE_DIR/'fig5.png')

# -------------------------------------------------------
# | Okay, let us come back to the seasonality, and Fourier decompositions.
# | Let us take 10 frequencies.

fourier = CalendarFourier(freq="A", order=10)

dp = DeterministicProcess(
    index=tunnel.index,
    constant=True,               # dummy feature for bias (y-intercept)
    order=1,                     # trend (order 1 means linear)
    seasonal=True,               # weekly seasonality (indicators)
    additional_terms=[fourier],  # annual seasonality (Fourier)
    drop=True,                   # drop terms to avoid collinearity
)

X = dp.in_sample()  # create features for the dates in tunnel.index
display(X.head(3))

y = tunnel['NumVehicles']


# -------------------------------------------------------
# | Here actual modeling.

model = LinearRegression()
model.fit(X, y)
y_fit = model.predict(X)

# -------------------------------------------------------
# | Show the results.

trace_1 = go.Scatter(x=X.index.to_timestamp(),
                     y=y,
                     line=dict(color='teal'),
                     mode='lines')

trace_2 = go.Scatter(x=X.index.to_timestamp(),
                     y=y_fit,
                     line=dict(color='coral'),
                     mode='lines')

data = [trace_1, trace_2]

layout = go.Layout(height=512, width=2048,
                   font=dict(size=20),
                   showlegend=False)

fig = go.Figure(data=data, layout=layout)
fig.show()
fig.write_image(IMAGE_DIR/'fig6.png')

# -------------------------------------------------------
# | Training error.

train_rmse = mean_squared_error(y, y_fit, squared=False)
print(f'RMSE : \033[96m{train_rmse:6.2f}\033[0m')

# -------------------------------------------------------
# | END
