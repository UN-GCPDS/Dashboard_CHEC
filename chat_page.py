
import dash
from app import app
import dash_bootstrap_components as dbc
from dash import html, dcc, Output, Input, callback_context, exceptions
from dash.dependencies import Input, Output, State, ALL

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

import json
import base64
import uuid
from flask import send_from_directory
import io
import PyPDF2  # Asegúrate de importar PyPDF2 al inicio
import pickle
import time
import maps_page

from functions.utils import load_previous_conversations, save_conversations, conversation, load_structured_data, update_documents_procces, recomendacion_apoyos, get_recommendations

from ui_components.ui_chat import create_layout

# Definición de estilos base
FONT_FAMILY = '''-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Ubuntu, "Helvetica Neue", Helvetica, Arial, "PingFang SC",
    "Hiragino Sans GB", "Microsoft Yahei UI", "Microsoft Yahei",
    "Source Han Sans CN", sans-serif'''

layout = create_layout() # type: ignore

# Definir estilos personalizados
ESTILOS = {
    'boton_principal': {
        'backgroundColor': '#80A022',
        'color': 'white',
        'border': 'none',
        'width': '100%'  # Simula block=True
    },
    'boton_secundario': {
        'backgroundColor': '#00782B',
        'color': 'white',
        'border': 'none',
        'width': '100%'  # Simula block=True
    },
    'boton_enviar': {  # Nuevo estilo para el botón Enviar
        'backgroundColor': '#80A022',
        'color': 'white',
        'border': 'none',
        'width': 'auto'  # Permite que el botón tenga su ancho natural
    },
    'fondo_lateral': {
        'backgroundColor': '#F0F6E7'
    },
    'fondo_principal': {
        'backgroundColor': 'white'
    },
    'texto_titulo': {
        'color': '#00782B'
    },
    'entrada_usuario': {
        'borderColor': '#80A022'
    },
    'mensaje_usuario': {
        'textAlign': 'right',
        'color': '#FFFFFF',
        'padding': '10px',
        'backgroundColor': '#067325',
        'borderRadius': '15px',
        'maxWidth': '70%',
        'alignSelf': 'flex-end',
        'margin': '5px',
        'fontSize': '19px',
        'white-space': 'pre-wrap',
    },
    'mensaje_asistente': {
        'textAlign': 'left',
        'color': '#FFFFFF',
        'padding': '10px',
        'backgroundColor': '#139d22',
        'borderRadius': '15px',
        'maxWidth': '70%',
        'alignSelf': 'flex-start',
        'margin': '5px',
        'fontSize': '19px',
        'white-space': 'pre-wrap',
    },
    'logo_chec': {
        'width': '200px',          # Aumenta el ancho a 200 píxeles
        'height': 'auto',          # Mantiene la proporción
        'marginBottom': '20px'     # Espaciado debajo del logo
    },
    'upload_card': {
        'backgroundColor': '#F0F6E7',  # Coincide con 'fondo_lateral'
        'border': '1px solid #00782B',  # Bordes en color secundario
        'borderRadius': '10px',
        'padding': '20px',
        'marginBottom': '20px'
    },
    'upload_header': {
        'color': '#00782B',  # Título en color secundario
        'textAlign': 'center',
        'marginBottom': '10px',
        'fontWeight': 'bold'  # Añadido para resaltar el título
    },
    'upload_dropzone': {
        'width': '100%',
        'height': '70px',
        'lineHeight': '70px',
        'borderWidth': '2px',
        'borderStyle': 'dashed',
        'borderRadius': '10px',
        'backgroundColor': '#E6F4EA',  # Coincide con 'mensaje_usuario'
        'color': '#00782B',
        'textAlign': 'center',
        'cursor': 'pointer'
    },
    'upload_dropdown': {
        'width': '100%',
        # 'height': '70px',  # Eliminada para evitar apachurramiento
        'borderRadius': '10px',
        'borderColor': '#80A022',  # Coincide con 'entrada_usuario'
        'backgroundColor': 'white',
        'color': '#00782B',
        # Añadimos padding para mejorar la legibilidad
        'padding': '5px'
    },
    'upload_status': {
        'marginTop': '10px'
    },
    'titulo_upload': {
        'color': '#00782B',
        'fontWeight': 'bold',
        'textAlign': 'center',
        'marginBottom': '15px',
        'fontSize': '1.2em'  # Ajuste de tamaño de fuente para mayor visibilidad
    }
}
# Lista de opciones para los Dropdowns de proceso
PROCESOS = [
    {"label": "General", "value": "general"},
    {"label": "Disposiciones Generales RETIE", "value": "capitulo_1"},
    {"label": "Productos Objetos RETIE", "value": "capitulo_2"},
    {"label": "Instalaciones Objeto RETIE", "value": "capitulo_3"},
    {"label": "Evaluacion de la Conformidad RETIE", "value": "capitulo_4"},
    {"label": "Resolución 40117", "value": "resolucion_40117"},
    {"label": "Normativa Apoyos", "value": "normativa_apoyos"},
    {"label": "Normativa Protecciones", "value": "normativa_protecciones"},
    {"label": "Normativa Aisladores", "value": "normativa_aisladores"},
    {"label": "Normativa Redes Aereas", "value": "redes_aereas_media_tension"},
    {"label": "Código Eléctrico Colombiano", "value": "codigo_electrico_colombiano"},
    {"label": "Requisitos Redes Aéreas", "value": "requisitos_redes_aereas"},
    {"label": "RETIE", "value": "retie"},
    {"label": "Interrupciones", "value": "interrrupciones_transformadores"},
    {"label": "Generar Gráficos", "value": "generate_plots"},
    {"label": "Recomendación", "value": "recomendacion"},
]

# Crear una lista sin "General" para la carga de archivos
PROCESOS_UPLOAD = [proceso for proceso in PROCESOS if proceso["value"] != "general"]

# Crear la carpeta para archivos subidos si no existe
UPLOAD_DIRECTORY = "Unstructured_Files"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


if maps_page.criticidad_data is not None:
    
    get_recommendations(maps_page.criticidad_data)

    
# Ruta para servir archivos desde 'Unstructured_Files'
@app.server.route('/Unstructured_Files/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory(UPLOAD_DIRECTORY, filename)

# Ruta para servir archivos desde 'plots'
@app.server.route('/plots/<path:filename>')
def serve_plot_file(filename):
    return send_from_directory('plots', filename)

@app.callback(
    Output('chat-data', 'data'),
    Output('entrada-usuario', 'value'),
    Input('nuevo-chat', 'n_clicks'),
    Input('enviar-btn', 'n_clicks'),
    Input('entrada-usuario', 'n_submit'),  # Agregar n_submit como Input
    Input({'type': 'chat-boton', 'index': ALL}, 'n_clicks'),
    State('entrada-usuario', 'value'),
    State('chat-data', 'data'),
    State('model-select', 'value'),  # Obtener el modelo seleccionado
    State('proceso-select', 'value'),  # Obtener el proceso seleccionado
    prevent_initial_call=True
)
def manejar_chat(n_clicks_nuevo, n_clicks_enviar, n_submit, n_clicks_chat, mensaje_usuario, data, modelo_seleccionado, proceso_seleccionado):
    ctx = callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered = ctx.triggered[0]
    prop_id = triggered['prop_id'].split('.')[0]
    prop_sub_id = triggered['prop_id'].split('.')[1] if '.' in triggered['prop_id'] else ''

    # Inicializar variables de salida
    nueva_data = data.copy()
    nuevo_valor_entrada = ''

    if prop_id == 'nuevo-chat':
        if n_clicks_nuevo:
            new_chat_id = f'chat-{len(data["chats"])}'
            print("NEW CHAT ID",new_chat_id)
            nueva_data['chats'][new_chat_id] = {'nombre': None, 'mensajes': [], 'files': []}
            nueva_data['current_chat_id'] = new_chat_id
            # Guardar las conversaciones actualizadas
            save_conversations(nueva_data)

    elif prop_id == 'enviar-btn' or (prop_id == 'entrada-usuario' and prop_sub_id == 'n_submit'):
        if nueva_data.get('current_chat_id') and mensaje_usuario:
            chat_id = nueva_data['current_chat_id']
            # Agregar mensaje del usuario y marcar que necesita respuesta
            mensaje_user = {
                'autor': 'Tú',
                'texto': mensaje_usuario,
                'needs_response': True,
                'modelo': modelo_seleccionado,      # Guardar el modelo seleccionado
                'proceso': proceso_seleccionado     # Guardar el proceso seleccionado
            }
            nueva_data['chats'][chat_id]['mensajes'].append(mensaje_user)

            # Si es el primer mensaje del usuario, asignar el nombre del chat
            if nueva_data['chats'][chat_id]['nombre'] is None:
                words = mensaje_usuario.split()
                topic = ' '.join(words[:5]) if len(words) >= 5 else mensaje_usuario
                nueva_data['chats'][chat_id]['nombre'] =  chat_id  # Puedes personalizar el nombre según tu lógica

            nuevo_valor_entrada = ''

            # Guardar las conversaciones actualizadas
            save_conversations(nueva_data)

    else:
        # Manejar los botones de chat
        try:
            button_id = json.loads(prop_id)
            if button_id['type'] == 'chat-boton':
                nueva_data['current_chat_id'] = button_id['index']
                # Guardar las conversaciones actualizadas
                save_conversations(nueva_data)
        except json.JSONDecodeError:
            pass  # Manejar IDs no JSON si es necesario

    return nueva_data, nuevo_valor_entrada

@app.callback(
    [Output('chats-history', 'style'),
     Output('chat-window', 'style'),
     Output('toggle-button', 'style'),
     Output('toggle-state', 'data')],
    [Input('toggle-button', 'n_clicks')],
    [State('toggle-state', 'data')]
)

def toggle_divs(n_clicks, toggle_state):

    chats_history_style = [{
            'height': '99.9%',
            'width': '25%',
            'overflowY': 'auto',
            'backgroundColor': '#F0F6E7',
            'transition': 'height 0.5s ease'
            },
            {
            'height': '99.9%',
            'width': '0%',
            'overflowY': 'auto',
            'backgroundColor': '#F0F6E7',
            'transition': 'height 0.5s ease'
            }]
    
    chat_window_style = [{
            'display': 'flex',
            'flexDirection': 'row',
            'width': '75%',
            'height': '100%',
            'backgroundColor': '#f0f6e7',
            'transition': 'height 0.5s ease'
            },
            {
            'display': 'flex',
            'flexDirection': 'row',
            'width': '100%',
            'height': '100%',
            'backgroundColor': '#f0f6e7',
            'transition': 'height 0.5s ease'
            }]
    
    toggle_button_style = [{
            'width': '3vh',
            'height': '12%',
            'border': 'none',
            'margin': '20vh 0 0 0',
            'backgroundColor': '#f0f6e7',
            'backgroundImage': "url('/assets/images/left-arrow-next-svgrepo-com.svg')",
            'backgroundSize': '100%',
            'backgroundPosition': 'center',
            'backgroundRepeat': 'no-repeat',
            'cursor': 'pointer',
            'transition': 'height 0.5s ease'
            },
            {
            'width': '3vh',
            'height': '12%',
            'border': 'none',
            'margin': '20vh 0 0 0',
            'backgroundColor': '#f0f6e7',
            'backgroundImage': "url('/assets/images/left-arrow-next-svgrepo-com.svg')",
            'backgroundSize': '100%',
            'backgroundPosition': 'center',
            'backgroundRepeat': 'no-repeat',
            'cursor': 'pointer',
            'transition': 'height 0.5s ease',
            'transform': 'rotate(180deg)'
            }]

    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # Alternar el estado actual
    toggle_state = not toggle_state

    # Estilos según el estado
    chats_history = chats_history_style[0] if toggle_state else chats_history_style[1]
    chat_window = chat_window_style[0] if toggle_state else chat_window_style[1]
    toggle_button = toggle_button_style[0] if toggle_state else toggle_button_style[1]

    return chats_history, chat_window, toggle_button, toggle_state


# Callback para generar la respuesta del asistente
@app.callback(
    Output('chat-data', 'data', allow_duplicate=True),
    Input('chat-data', 'data'),
    prevent_initial_call=True
)
def generar_respuesta_asistente(data):
    chat_id = data.get('current_chat_id')
    if not chat_id:
        raise dash.exceptions.PreventUpdate

    mensajes = data['chats'][chat_id]['mensajes']

    if not mensajes:
        raise dash.exceptions.PreventUpdate

    last_message = mensajes[-1]

    if last_message['autor'] == 'Tú' and last_message.get('needs_response', False):
        # Obtener el modelo y proceso seleccionados para este mensaje
        modelo = last_message.get('modelo', 'gpt')  # Valor por defecto 'gpt' si no está definido
        proceso = last_message.get('proceso', 'general')  # Obtener el proceso seleccionado


        # Generar la respuesta del asistente utilizando el modelo seleccionado
        mensaje_usuario = last_message['texto']
        response_llm, flag_image = conversation(chat_id, mensaje_usuario, modelo, proceso)  # Pasar el modelo y proceso

        if flag_image:
            time.sleep(5)
            number_image=int(len(os.listdir(f"plots/{chat_id}")))-1
            path_plot=f"plots/{chat_id}/output_{number_image}.jpg"
            print(path_plot)
            # Crear una respuesta que incluya la imagen y el texto
            respuesta_asistente = {
                'autor': 'Asistente',
                'imagen': path_plot,  # Ruta relativa a la carpeta assets
                'texto': "¿Hay algo más en lo que te pueda ayudar?"
            }
        else:
            # Crear una respuesta solo con el texto
            respuesta_asistente = {'autor': 'Asistente', 'texto': response_llm}

        # Agregar la respuesta del asistente a los mensajes del chat
        data['chats'][chat_id]['mensajes'].append(respuesta_asistente)
        # Marcar que el mensaje del usuario ya tiene respuesta
        last_message['needs_response'] = False
        # Guardar las conversaciones actualizadas
        save_conversations(data)
        return data
    else:
        raise dash.exceptions.PreventUpdate

# Callback para actualizar la lista de chats con nombres dinámicos
@app.callback(
    Output('lista-chats', 'children'),
    Input('chat-data', 'data')
)
def actualizar_lista_chats(data):
    chats = data.get('chats', {})
    children = []
    
    # Invertir el orden de los chats para mostrar el más reciente primero
    for idx, (chat_id, chat_info) in enumerate(reversed(list(chats.items())), 1):
        # Determinar el nombre del chat
        nombre = chat_info['nombre'] if chat_info['nombre'] else f'Chat {idx}'
        boton_chat = dbc.Button(
            nombre,
            id={'type': 'chat-boton', 'index': chat_id},
            style=ESTILOS['boton_secundario'],
            className='mb-1',
            n_clicks=0,
        )
        children.append(boton_chat)
    return children

# Callback para actualizar la ventana del chat cuando los datos cambian
@app.callback(
    Output('ventana-chat', 'children'),
    Input('chat-data', 'data')
)
def actualizar_ventana_chat(data):
    chat_id = data.get('current_chat_id')
    if chat_id is None:
        return []
    mensajes = data['chats'].get(chat_id, {}).get('mensajes', [])
    contenido = []

    # Agregar mensajes de chat
    for msg in mensajes:
        if 'texto' in msg and 'imagen' in msg:
            # Mensaje que contiene tanto imagen como texto
            estilo_imagen = {
                'maxWidth': '70%',
                'borderRadius': '15px',
                'margin': '5px'
            }
            estilo_texto = ESTILOS['mensaje_asistente']  # Puedes ajustar el estilo si es necesario

            contenido.append(html.Div(
                [
                    html.Img(src=msg['imagen'], style=estilo_imagen, alt='Imagen generada'),
                    html.P(f"{msg['texto']}", style=estilo_texto)
                ],
                style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'flex-start'}
            ))
        elif 'texto' in msg:
            estilo_mensaje = ESTILOS['mensaje_usuario'] if msg['autor'] == 'Tú' else ESTILOS['mensaje_asistente']
            contenido.append(html.Div(
                html.P(f"{msg['texto']}", style=estilo_mensaje),
                style={'display': 'flex', 'justifyContent': 'flex-end' if msg['autor'] == 'Tú' else 'flex-start', 'white-space': 'pre-wrap'}
            ))
        elif 'imagen' in msg:
            # Estilo para la imagen del asistente
            estilo_imagen = {
                'maxWidth': '70%',
                'borderRadius': '15px',
                'margin': '5px'
            }
            contenido.append(html.Div(
                html.Img(src=msg['imagen'], style=estilo_imagen, alt='Imagen generada'),
                style={'display': 'flex', 'justifyContent': 'flex-start'}
            ))

    return contenido

# Callback para manejar la carga de archivos PDF y la selección de proceso
@app.callback(
    Output('upload-status', 'children'),
    Input('upload-pdf', 'contents'),
    State('upload-pdf', 'filename'),
    State('upload-proceso-select', 'value')
)
def handle_file_upload(contents, filename, proceso_seleccionado):
    if contents is not None:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            print(f"Archivo decodificado correctamente: {filename}")
        except Exception as e:
            print(f"Error al decodificar el archivo: {e}")
            return dbc.Alert("Error al decodificar el archivo.", color="danger", dismissable=True)

        # Verificar que el archivo sea un PDF
        if not filename.lower().endswith('.pdf'):
            return dbc.Alert("Solo se permiten archivos PDF.", color="danger", dismissable=True)

        # Verificar que un proceso haya sido seleccionado
        if not proceso_seleccionado:
            return dbc.Alert("Por favor, selecciona un proceso antes de subir el archivo.", color="warning", dismissable=True)

        # Extraer la extensión del archivo
        name, ext = os.path.splitext(filename)

        # Reemplazar espacios y caracteres no permitidos en el nombre del proceso
        proceso_sanitizado = "".join(c if c.isalnum() else "_" for c in proceso_seleccionado)

        # Generar un nombre único incorporando el proceso
        unique_filename = f"{proceso_sanitizado}_{uuid.uuid4()}{ext}"
        file_path = os.path.join(UPLOAD_DIRECTORY, unique_filename)

        # Validar el PDF directamente desde la memoria
        try:
            pdf_stream = io.BytesIO(decoded)
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            num_pages = len(pdf_reader.pages)
            print(f"PDF válido. Número de páginas: {num_pages}")
        except Exception as e:
            print(f"Validación de PDF fallida: {e}")
            return dbc.Alert("El archivo subido no es un PDF válido.", color="danger", dismissable=True)
        
        
        # Guardar el archivo en el sistema de archivos
        try:
            with open(file_path, 'wb') as f:
                f.write(decoded)
            print(f"Archivo guardado correctamente en: {file_path}")
            update_documents_procces(proceso_sanitizado,file_path)

        except Exception as e:
            print(f"Error al guardar el archivo: {e}")
            return dbc.Alert(f"Error al subir el archivo: {str(e)}", color="danger", dismissable=True)

        # Asociar el archivo con el proceso seleccionado en los datos del chat
        data = load_previous_conversations()
        current_chat_id = data.get('current_chat_id')

        if current_chat_id:
            if 'files' not in data['chats'][current_chat_id]:
                data['chats'][current_chat_id]['files'] = []
            data['chats'][current_chat_id]['files'].append({
                'filename': filename,
                'filepath': f"Unstructured_Files/{unique_filename}",  # Ruta relativa para el enlace
                'proceso': proceso_seleccionado
            })
            save_conversations(data)

        return dbc.Alert(
            f"Archivo '{filename}' subido exitosamente para el proceso '{proceso_seleccionado}'.",
            color="success",
            dismissable=True
        )
    return ""

@app.callback(
    Output('url-chat', 'pathname'),
    [Input('button-maps', 'n_clicks'),
     Input('button-graphs', 'n_clicks'),
     Input('button-tab-net', 'n_clicks')]
)
def redirect_to_pages(n_clicks_maps, n_clicks_graphs, n_clicks_tab_net):
    # Obtener el contexto del trigger
    ctx = callback_context
    if not ctx.triggered:
        raise exceptions.PreventUpdate

    # Identificar qué botón disparó el callback
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'button-maps' and n_clicks_maps:
        return "/maps_page"
    elif triggered_id == 'button-graphs' and n_clicks_graphs:
        return "/graphs_page"
    elif triggered_id == 'button-tab-net' and n_clicks_tab_net:
        return "/tab-net_page"

    raise exceptions.PreventUpdate
