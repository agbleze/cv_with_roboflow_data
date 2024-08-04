
#%%
from dash import html, Input, Output, State, dcc
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import requests
import json
from dash.exceptions import PreventUpdate



app = dash.Dash(__name__, 
                external_stylesheets=[
                                                dbc.themes.SOLAR,
                                                dbc.icons.BOOTSTRAP,
                                                dbc.icons.FONT_AWESOME,
                                            ]
                )

#app.layout = prediction_layout

#app.validation_layout = [prediction_layout]



app.run_server(port='4048', host='0.0.0.0', debug=False, use_reloader=False)
# %%
