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


'''# Ruta a la carpeta donde están los archivos
carpeta = './memories'

# Listar todos los archivos en la carpeta
archivos = os.listdir(carpeta)

# Iterar sobre los archivos
for archivo in archivos:
    # Verificar si no es 'chat-0.pkl'
    if archivo != 'chat-0.pkl' and archivo.startswith('chat-') and archivo.endswith('.pkl'):
        # Construir la ruta completa al archivo
        ruta_archivo = os.path.join(carpeta, archivo)
        # Eliminar el archivo
        os.remove(ruta_archivo)
        print(f"Eliminado: {ruta_archivo}")

os.remove('chat_data.json')

# Datos a escribir en el archivo JSON
chat_data = {
    "chats": {
        "chat-0": {
            "nombre": "chat-0",
            "mensajes": [
                {
                    "autor": "Tú",
                    "texto": "",
                    "needs_response": False,
                    "modelo": "gpt",
                    "proceso": "recomendacion"
                },
                {
                    "autor": "Asistente",
                    "texto": "Hola, soy un asistente para ayudarte con tus consultas. ¿En qué puedo ayudarte hoy?"
                }
            ],
            "files": []
        }
    },
    "current_chat_id": "chat-0"
}

with open('chat_data.json', 'w', encoding='utf-8') as f:
    json.dump(chat_data, f, indent=4, ensure_ascii=False)

print("Datos de la sesión anterior con el chat limpiados.")'''


from app import app
import maps_page
import chat_page
import graphs_page
import tab_net_page

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content',style={'border': 'None'})
])

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

if __name__ == '__main__':
# Abrir el archivo en modo lectura

    app.run_server(debug=True)