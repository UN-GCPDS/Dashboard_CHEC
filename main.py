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
# Abrir el archivo en modo lectura

    with open("C:/Users/lucas/OneDrive - Universidad Nacional de Colombia/PC-GCPDS/Documentos/data/OPENAI_API_Key.txt", "r") as archivo:
        OPENAI_API_KEY = archivo.read()

    os.environ['OPENAI_API_KEY'] = "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"
    app.run_server(debug=True)