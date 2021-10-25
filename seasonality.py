# |------------------------------------------------------------------
# | # Tunnel Traffic  -- Time Sequence Analysis T1
# |------------------------------------------------------------------
# |
# | ## 1. Introduction
# |
# | This is a notebook to practice routine procedures
# | in the time sequence analysis.
# |
# | Temporal sequence consists of several components.
# |
# | - trend (gradual decrease / increase)
# | - seaonality (hour of day, day of week, week of month, month of year)
# | - cycle (up and down but with a specific time scale)
# | - pecuriality (national holidays)
# |
# | In a sense these are the only elements that one can predict
# | by mahcine learning, and nothign else.
# | A  model can predict  what only  repeats. What happened before.
# | A model cannot learn from what did not happen.
# |
# | To deal with each elemetns of time sequence, we have
# |
# | - __trend__ : Analytical fitting of baseline (lienar, polynomial, etc)
# | - __seasonality__ : Fourier decomposition
# | - __cycle__ : Lags
# | - __pecuriality__ : Categorical featuers
# |
# | In this notebook we will familarize ourselve with
# |
# |  * Manupulation of index.
# |  * 'DeterministicProcess` in 'statsmodels' package as 'time dummy'.
# |
# | Index in other types of data is just convenient unique address
# | of a record to identify it. Since indices  do not contain
# | useful information, they are usually dropped in the machine learning.
# | On the contray, index has special meaning in the time sequence anlysis.
# | It is the feature needed for the prediction.
# | The manupulatino of index in time sequence analysis include

# | - Date parsing
# | - Set date as index
# | - Set period of date to 'D' (or 'M', 'W )
# |   + (offset aliases)[https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases]
# | - Set `dtype` (category features to `category`)
# |
# | The only features (except categorical ones) used to model the temporal
# | behaviour of the target is time. But time in different intervals.
# | 'DeterministicProcess` is used to quickly create `t` in `y=f(t)`.
# | It is called 'deterministic', because it is a feature that we do not
# | have a control. One can use such features at the time of prediction, i.e.
# | if we would like to predict a sales on a sunday, we can use the fact
# | that day is a sunday. On the contrary, we cannot use a sales one day before
# | if it is not published yet at the time the prediction is to be executed.
# | The former (=being a Sunday) is a dterministic feature, while the sales on
# | Saturday is a non-deterministic feature.
# |
# | ## 2. Task
# |
# |   We have a record of trafic in a tunnel. Model the temporal sequence of
# | the trafic from the time features only.

# | ## 3. Data
# |
# | 1. A traffic of vehicles traveling through the Baregg Tunnel
# | in Switzerland each day from November 2003 to November 2005.
#
# | ## 3. Notebook
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

from kaggle_tsa.ktsa import *
from IPython.display import display
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import kaleido


# -------------------------------------------------------
# | Set up directories.
CWD = Path('/Users/meg/git7/trend/')
DATA_DIR = Path('../input/ts-course-data/')
KAGGLE_DIR = Path('ryanholbrook/ts-course-data/')
IMAGE_DIR = Path('./images')
HTML_DIR = Path('./html')

os.chdir(CWD)
set_cwd(CWD)

set_data_dir(KAGGLE_DIR, CWD)
show_whole_dataframe(True)

# -------------------------------------------------------
# | Read the data, first as it is.

tunnel = pd.read_csv(DATA_DIR/'tunnel.csv')

# | Check the contents.

print(tunnel.info())
display(tunnel.head(3))

# | Index manipulation. Parsing. Currently `Day` column is taknen as
# | `object` type (=string). We will convert it to `datetime` type.
# | In the same time `Day` column is set to be the index.
# | The frequency of the index is now set to 'D' (=calendar day frequency).
# | By setting the frequency, when we use `.shift(1)` to the dataframe,
# | the dataframe will be shifted by that amount in the unit of specified
# | freqency. The `dtype`  of `NumVehicles` is set to 'float32'
# |

tunnel = pd.read_csv(DATA_DIR/'tunnel.csv',
                     dtype={'NumVehicles': 'float32'},
                     parse_dates=['Day'],
                     index_col='Day').to_period('D')

print(tunnel.info())
print(tunnel.index)
display(tunnel.head(3))

# -------------------------------------------------------
# | Let us have a quick look af the data.

trace_1 = go.Scatter(x=tunnel.index.to_timestamp(),
                     y=tunnel['NumVehicles'],
                     name='Tunnel traffic')

data = [trace_1]
layout = go.Layout(height=512)
fig = go.Figure(data=data, layout=layout)
embed_plot(fig, HTML_DIR/'p_1.html')
fig.show()
# fig.write_image(IMAGE_DIR/'fig1.png')

# -------------------------------------------------------
# | One can see
# |
# | - Weekly trend (in-week common trend).
# | - In-year trend. Decrease of traffic  during
# |    + Christman holidays
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

# -------------------------------------------------------
# | The trend is nicely seen that
# | the trafic gradually goes up from Monday to Friday,
# | and decreases toward Saturday and Sunday.
# -------------------------------------------------------
# =======================================================
# | we will make two more pltos.
# | 1. periodogram
# | 2. lagplot

# - check if index is continuous and no missing date
# =======================================================

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

trace = go.Scatter(x=frequencies,
                   y=spectrum,
                   fill='tozeroy',
                   #                   fillcolor='coral',
                   line=dict(color='coral'),
                   line_shape='hvh')

data = [trace]
layout = go.Layout(height=640,
                   font=dict(size=20),
                   xaxis=dict(type='log',
                              ticktext=annual_freq,
                              tickvals=[1, 2, 4, 6, 12, 26, 52, 104]))
#                              categoryorder='array')
#                              categoryarray=annual_freq,
#                             categoryorder='array'))

fig = go.Figure(data=data, layout=layout)
fig.show()

# | detrend {'linear', 'constant'}
# | window  {'boxcar', 'gaussian', ...} `scipy.signal.get_window`
# | [shape of window functon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window)
# | scaling {'density', 'spectrum'} [V^2/Hz] (power spectrum) [V^2]

# -------------------------------------------------------
# | We can see
# | * strong weekly repetitions, and
# | * annual, semiannual, and quaterly trend.
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

corr = [tunnel['NumVehicles'].autocorr(lag=i) for i in range(1, n_lag+1)]

_ = [print(f'Lag {i:-2}: {tunnel["NumVehicles"].autocorr(lag=i):5.3f}')
     for i in range(1, n_lag+1)]

# -------------------------------------------------------
# | There are 3 components seen in Lag 1 plot.
# |
# | - a bit increased from the day before
# | - a bit dereased from the day before
# | - about the same from the day before
# |
# | There are 3 components seen in Lag 1 plot.
# | The correlation is higher in 7 days lag than 1 day lat.
# | It corresponds to the weekly repetitions of the traffic.
# |

# -------------------------------------------------------
# Partial autocorrelation function.

fig = plot_pacf(tunnel['NumVehicles'], lags=12)
fig.show()

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

# -------------------------------------------------------
# | Let us take 10 frequencies of Fourier decompositions

fourier = CalendarFourier(freq="A", order=10)

dp = DeterministicProcess(
    index=tunnel.index,
    constant=True,               # dummy feature for bias (y-intercept)
    order=1,                     # trend (order 1 means linear)
    seasonal=True,               # weekly seasonality (indicators)
    additional_terms=[fourier],  # annual seasonality (fourier)
    drop=True,                   # drop terms to avoid collinearity
)

X = dp.in_sample()  # create features for dates in tunnel.index
display(X.head(3))

y = tunnel['NumVehicles']
model = LinearRegression()
model.fit(X, y)
y_fit = model.predict(X)

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
# # -------------------------------------------------------
# | Training

train_rmse = mean_squared_error(y, y_fit, squared=False)
print(f'RMSE : \033[96m{train_rmse:6.2f}\033[0m')


# # -------------------------------------------------------
# # 10 sin/cos pairs for "A"nnual seasonality
# # Let's take a look at seasonal plots over a week and over a year.
# # X = tunnel.copy()
# # # days within a week
# # X["day"] = X.index.dayofweek  # the x-axis (freq)
# # X["week"] = X.index.week  # the seasonal period (period)

# # # days within a year
# # X["dayofyear"] = X.index.dayofyear
# # X["year"] = X.index.year
# # fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 6))
# # seasonal_plot(X, y="NumVehicles", period="week", freq="day", ax=ax0)
# # seasonal_plot(X, y="NumVehicles", period="year", freq="dayofyear", ax=ax1)


# # lagplot

# # plot_periodogram(tunnel.NumVehicles)


# # The periodogram agrees with the seasonal plots above: a strong weekly season and a weaker annual season. The weekly season we'll model with indicators and the annual season with Fourier features. From right to left, the periodogram falls off between *Bimonthly (6)* and *Monthly (12)*, so let's use 10 Fourier pairs.
# #
# # We'll create our seasonal features using `DeterministicProcess`, the same utility we used in Lesson 2 to create trend features. To use two seasonal periods (weekly and annual), we'll need to instantiate one of them as an "additional term":

# # In[6]:


# # With our feature set created, we're ready to fit the model and make predictions. We'll add a 90-day forecast to see how our model extrapolates beyond the training data. The code here is the same as that in earlier lessons.

# # In[7]:


# y = tunnel["NumVehicles"]

# model = LinearRegression(fit_intercept=False)
# _ = model.fit(X, y)

# y_pred = pd.Series(model.predict(X), index=y.index)
# X_fore = dp.out_of_sample(steps=90)
# y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

# ax = y.plot(color='0.25', style='.',
#             title="Tunnel Traffic - Seasonal Forecast")
# ax = y_pred.plot(ax=ax, label="Seasonal")
# ax = y_fore.plot(ax=ax, label="Seasonal Forecast", color='C3')
# _ = ax.legend()


# ---
#
# There's still more we can do with time series to improve our forecasts. In the next lesson, we'll learn how to use time series themselves as a features. Using time series as inputs to a forecast lets us model the another component often found in series: *cycles*.
#
# # Your Turn #
#
# [**Create seasonal features for Store Sales**](https://www.kaggle.com/kernels/fork/19615991) and extend these techniques to capturing holiday effects.


# Knowing how Fourier features are computed isn't essential to using them, but if seeing the details would clarify things, the cell hidden cell below illustrates how a set of Fourier features could be derived from the index of a time series. (We'll use a library function from `statsmodels` for our applications, however.)

# # In[1]:


# def fourier_features(index, freq, order):
#     time = np.arange(len(index), dtype=np.float32)
#     k = 2 * np.py * (1/freq) * time
#     features = {}
#     for i in range(1, order+1):
#         features.update({
#             f'sin_{freq}_{i}': np.sin(i*k),
#             f'cos_{freq}_{i}': np.cos(i*k),
#         })
#     return pd.DataFrame(features, index=index)


# # In[2]:


# def fourier_features(index, freq, order):
#     time = np.arange(len(index), dtype=np.float32)
#     k = 2 * np.pi * (1 / freq) * time
#     features = {}
#     for i in range(1, order + 1):
#         features.update({
#             f"sin_{freq}_{i}": np.sin(i * k),
#             f"cos_{freq}_{i}": np.cos(i * k),
#         })
#     return pd.DataFrame(features, index=index)


# # Compute Fourier features to the 4th order (8 new features) for a
# # series y with daily observations and annual seasonality:
# #
# # fourier_features(y, freq=365.25, order=4)


# # Example - Tunnel Traffic #
# #
# # We'll continue once more with the *Tunnel Traffic* dataset. This hidden cell loads the data and defines two functions: `seasonal_plot` and `plot_periodogram`.

# # In[ ]:


# # In[3]:


# simplefilter("ignore")

# # Set Matplotlib defaults
# plt.style.use("seaborn-whitegrid")
# plt.rc("figure", autolayout=True, figsize=(11, 5))
# plt.rc(
#     "axes",
#     labelweight="bold",
#     labelsize="large",
#     titleweight="bold",
#     titlesize=16,
#     titlepad=10,
# )
# plot_params = dict(
#     color="0.75",
#     style=".-",
#     markeredgecolor="0.25",
#     markerfacecolor="0.25",
#     legend=False,
# )
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

# # annotations: https://stackoverflow.com/a/49238256/5769929
# def seasonal_plot(X, y, period, freq, ax=None):
#     if ax is None:
#         _, ax = plt.subplots()
#     palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
#     ax = sns.lineplot(
#         x=freq,
#         y=y,
#         hue=period,
#         data=X,
#         ci=False,
#         ax=ax,
#         palette=palette,
#         legend=False,
#     )
#     ax.set_title(f"Seasonal Plot ({period}/{freq})")
#     for line, name in zip(ax.lines, X[period].unique()):
#         y_ = line.get_ydata()[-1]
#         ax.annotate(
#             name,
#             xy=(1, y_),
#             xytext=(6, 0),
#             color=line.get_color(),
#             xycoords=ax.get_yaxis_transform(),
#             textcoords="offset points",
#             size=14,
#             va="center",
#         )
#     return ax

# # from  scipy.signal import get_window
# # =======================================================


# def plot_periodogram(ts, detrend='linear', ax=None):
#     from scipy.signal import periodogram
#     fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
#     freqencies, spectrum = periodogram(
#         ts,
#         fs=fs,
#         detrend=detrend,
#         window="boxcar",
#         scaling='spectrum',
#     )
#     if ax is None:
#         _, ax = plt.subplots()
#     ax.step(freqencies, spectrum, color="purple")
#     ax.set_xscale("log")
#     ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
#     ax.set_xticklabels(
#         [
#             "Annual (1)",
#             "Semiannual (2)",
#             "Quarterly (4)",
#             "Bimonthly (6)",
#             "Monthly (12)",
#             "Biweekly (26)",
#             "Weekly (52)",
#             "Semiweekly (104)",
#         ],
#         rotation=30,
#     )
#     ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
#     ax.set_ylabel("Variance")
#     ax.set_title("Periodogram")
#     return ax


# data_dir = Path("../input/ts-course-data")
# tunnel = pd.read_csv(data_dir / "tunnel.csv", parse_dates=["Day"])
# tunnel = tunnel.set_index("Day").to_period("D")
