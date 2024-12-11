import dash
import dash_leaflet as dl
from dash import html, dcc, Input, Output, callback
import sys
import logging

# Configurar logging para asegurar que los mensajes se impriman
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Datos de ejemplo
ciudades = [
    {
        "nombre": "Londres", 
        "pos": [51.505, -0.09],
        "info": "Capital del Reino Unido, famosa por el Big Ben y el London Eye"
    },
    {
        "nombre": "París", 
        "pos": [48.8566, 2.3522],
        "info": "Ciudad del amor, hogar de la Torre Eiffel y el Louvre"
    },
    {
        "nombre": "Berlín", 
        "pos": [52.5200, 13.4050],
        "info": "Capital de Alemania, conocida por su rica historia y cultura moderna"
    }
]

app = dash.Dash(__name__)

app.layout = html.Div([
    # Mapa con marcadores y popups
    dl.Map([
        dl.TileLayer(),
        *[
            dl.Marker(
                position=ciudad['pos'], 
                children=[
                    dl.Tooltip(ciudad['nombre']),
                    dl.Popup([
                        html.H3(ciudad['nombre'], id=f'popup-title-{ciudad["nombre"].lower()}'),
                        html.P(ciudad['info'], id=f'popup-info-{ciudad["nombre"].lower()}')
                    ])
                ]
            ) for ciudad in ciudades
        ]
    ], 
    id='map',
    style={'width': '100%', 'height': '500px'},
    center=[50, 10],
    zoom=4
    ),
    
    # Área de depuración
    html.Div(id='debug-output')
])

# Callback para depuración con múltiples métodos de impresión
@app.callback(
    Output('debug-output', 'children'),
    Input('map', 'click_popup')
)
def depurar_popup(click_popup):
    # Método 1: sys.stderr
    sys.stderr.write("Método 1 (sys.stderr): Popup clickeado\n")
    sys.stderr.flush()

    # Método 2: logging
    logger.info("Método 2 (logging): Popup clickeado")

    # Método 3: print con flush
    print("Método 3 (print): Popup clickeado", flush=True)

    # Método 4: Imprimir contenido si está disponible
    if click_popup:
        try:
            # Intento de extraer y imprimir contenido
            contenido = str(click_popup)
            print("Contenido del popup:", contenido, flush=True)
            
            # Intento con acceso a propiedades
            if 'props' in click_popup and 'children' in click_popup['props']:
                for child in click_popup['props']['children']:
                    print("Hijo del popup:", child, flush=True)
        except Exception as e:
            print(f"Error al capturar popup: {e}", flush=True)

    return "Revisa la consola para los mensajes de depuración"

if __name__ == '__main__':
    # Añadir esta línea para forzar la salida
    import os
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    app.run_server(debug=True)