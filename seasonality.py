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
# | -------------------------------------
# | Import packages.

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

import plotly.graph_objs as go
from IPython.display import display

import os


import sklearn
sklearn.__version__

# | Set up directories.

DATA_DIR = Path('../input/ts-courses-data')
CWD

# | Read the data, first as it is.

tunnel = pd.read_csv(DATA_DIR/'tunnel.csv')

# | Check the contents.

print(tunnel.info())

display(tunnel.head(3))


# Knowing how Fourier features are computed isn't essential to using them, but if seeing the details would clarify things, the cell hidden cell below illustrates how a set of Fourier features could be derived from the index of a time series. (We'll use a library function from `statsmodels` for our applications, however.)

# In[1]:


def fourier_features(index, freq, order):
    time = np.arange(len(index), dtype=np.float32)
    k = 2 * np.py * (1/freq) * time
    features = {}
    for i in range(1, order+1):
        features.update({
            f'sin_{freq}_{i}': np.sin(i*k),
            f'cos_{freq}_{i}': np.cos(i*k),
        })
    return pd.DataFrame(features, index=index)


# In[2]:


def fourier_features(index, freq, order):
    time = np.arange(len(index), dtype=np.float32)
    k = 2 * np.pi * (1 / freq) * time
    features = {}
    for i in range(1, order + 1):
        features.update({
            f"sin_{freq}_{i}": np.sin(i * k),
            f"cos_{freq}_{i}": np.cos(i * k),
        })
    return pd.DataFrame(features, index=index)


# Compute Fourier features to the 4th order (8 new features) for a
# series y with daily observations and annual seasonality:
#
# fourier_features(y, freq=365.25, order=4)


# Example - Tunnel Traffic #
#
# We'll continue once more with the *Tunnel Traffic* dataset. This hidden cell loads the data and defines two functions: `seasonal_plot` and `plot_periodogram`.

# In[ ]:


# In[3]:


simplefilter("ignore")

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# annotations: https://stackoverflow.com/a/49238256/5769929
def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax


def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax


data_dir = Path("../input/ts-course-data")
tunnel = pd.read_csv(data_dir / "tunnel.csv", parse_dates=["Day"])
tunnel = tunnel.set_index("Day").to_period("D")


# Let's take a look at seasonal plots over a week and over a year.

# In[4]:


X = tunnel.copy()

# days within a week
X["day"] = X.index.dayofweek  # the x-axis (freq)
X["week"] = X.index.week  # the seasonal period (period)

# days within a year
X["dayofyear"] = X.index.dayofyear
X["year"] = X.index.year
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 6))
seasonal_plot(X, y="NumVehicles", period="week", freq="day", ax=ax0)
seasonal_plot(X, y="NumVehicles", period="year", freq="dayofyear", ax=ax1)


# Now let's look at the periodogram:

# In[5]:


plot_periodogram(tunnel.NumVehicles)


# The periodogram agrees with the seasonal plots above: a strong weekly season and a weaker annual season. The weekly season we'll model with indicators and the annual season with Fourier features. From right to left, the periodogram falls off between *Bimonthly (6)* and *Monthly (12)*, so let's use 10 Fourier pairs.
#
# We'll create our seasonal features using `DeterministicProcess`, the same utility we used in Lesson 2 to create trend features. To use two seasonal periods (weekly and annual), we'll need to instantiate one of them as an "additional term":

# In[6]:


# 10 sin/cos pairs for "A"nnual seasonality
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


# With our feature set created, we're ready to fit the model and make predictions. We'll add a 90-day forecast to see how our model extrapolates beyond the training data. The code here is the same as that in earlier lessons.

# In[7]:


y = tunnel["NumVehicles"]

model = LinearRegression(fit_intercept=False)
_ = model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=y.index)
X_fore = dp.out_of_sample(steps=90)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax = y.plot(color='0.25', style='.',
            title="Tunnel Traffic - Seasonal Forecast")
ax = y_pred.plot(ax=ax, label="Seasonal")
ax = y_fore.plot(ax=ax, label="Seasonal Forecast", color='C3')
_ = ax.legend()


# ---
#
# There's still more we can do with time series to improve our forecasts. In the next lesson, we'll learn how to use time series themselves as a features. Using time series as inputs to a forecast lets us model the another component often found in series: *cycles*.
#
# # Your Turn #
#
# [**Create seasonal features for Store Sales**](https://www.kaggle.com/kernels/fork/19615991) and extend these techniques to capturing holiday effects.
