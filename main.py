import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Output, Input
from dash.dependencies import Input, Output, State

import os
import random
import warnings
import folium
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from folium.plugins import MarkerCluster
from shapely.geometry import Point, Polygon, MultiPoint

from app import app
import maps_page
import chat_page
import graphs_page

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content',style={'border': 'None'})
])

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):   
    if pathname == '/chat_page':
        return chat_page.layout
    elif pathname == '/graphs_page':
        return graphs_page.layout
    else:
        return maps_page.layout

if __name__ == '__main__':
    OPENAI_API_KEY = "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"  # Reemplaza con tu clave de API
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    app.run_server(debug=True)