import pandas as pd
import geopandas as gpd

from pathlib import Path
import os
import webbrowser
import zipfile

import plotly.graph_objs as go
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster, HeatMap

# -------------------------------------------------------
# | Some housekeeping functions.
# | Later they will go to a module ``../kaggle_geospatial`.


def set_cwd(CWD):
    if Path.cwd() != CWD:
        os.chdir(CWD)

# | If we have not downloaded the course data, get it from Alexis Cook's
# | kaggle public dataset.


def set_data_dir(KAGGLE_DIR,  CWD):
    '''

    KAGGLE_DIR : Path() -  for the data source on the internet
    CWD : Path() - current working directory.

    This function assumes to be executed from CWD

    '''
    os.chdir(CWD)
    INPUT_DIR = Path('../input')

    if not INPUT_DIR.exists():
        INPUT_DIR.mkdir()
        os.chdir(INPUT_DIR)
        command = 'kaggle d download ' + str(KAGGLE_DIR)
        os.system(command)

        zip_list = list(Path('.').glob('*.zip'))

        for z in zip_list:
            z_path = Path(str(z).replace('.zip', ''))
            z_path.mkdir()
            z = z.rename(z_path/z)

            print(f'\033[31mz: {z}\033[0m')
            with zipfile.ZipFile(z, 'r') as zip_ref:
                zip_ref.extractall(z_path)

        os.chdir(str(CWD))

# | Some housekeeping stuff. Change `pandas`' options so that we can see
# | whole DataFrame without skipping the lines in the middle.


def show_whole_dataframe(show):
    if show:
        pd.options.display.max_rows = 999
        pd.options.display.max_columns = 99

# | This is to store the folium visualization to an html file, and show it
# | on the local browser.


def embed_map(m, file_name):
    from IPython.display import IFrame
    m.save(file_name)
    return IFrame(file_name, width='100%', height='500px')


def embed_plot(fig, file_name):
    from IPython.display import IFrame
    fig.write_html(file_name)
    return IFrame(file_name, width='100%', height='500px')


def show_on_browser(m, file_name):
    '''
    m   : folium Map object
    Do not miss the trailing '/'
    '''
    m.save(file_name)
    url = 'file://'+file_name
    webbrowser.open(url)

# -------------------------------------------------------
