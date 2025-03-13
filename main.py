import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Output, Input
from dash.dependencies import Input, Output, State

import os
import json
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
from utils.maps_functions import get_labels, my_mse_loss_fn, my_rmse_loss_fn, my_mae_loss_fn, my_mape_loss_fn, my_r2_score_fn, CustomTabNetRegressor
from app import app
import webbrowser
from threading import Thread
import time
import urllib.request
import maps_page
import chat_page
import graphs_page
import tab_net_page

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content',style={'border': 'None'})
])

import sys
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

app.config.suppress_callback_exceptions = True

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):   
    if pathname == '/chat_page':
        return chat_page.layout
    elif pathname == '/graphs_page':
        return graphs_page.layout
    elif pathname == '/tab-net_page':
        return tab_net_page.layout
    else:
        return maps_page.layout
    
def wait_and_open_browser():
    url = "http://127.0.0.1:8050"
    while True:
        try:
            urllib.request.urlopen(url)  # Intenta conectar
            webbrowser.open(url)  # Si conecta, abre el navegador
            break  # Salir del bucle cuando el servidor esté listo
        except:
            time.sleep(2)  # Espera 1 segundo antes de volver a intentar

if __name__ == '__main__':
    # Abrir el archivo en modo lectura
    # Iniciar el servidor en un hilo separado
    Thread(target=wait_and_open_browser, daemon=True).start()
    app.run_server(debug=True)