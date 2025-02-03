import os
import gc
import json
from app import app
import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import MATCH, ALL
from dash import callback_context, exceptions
from dash import Dash, html, dcc, Output, Input, State, ctx
import dash_bootstrap_components as dbc
from ui_components.ui_maps import create_layout
from utils.maps_functions import select_data, load_data, map_folium, enumerate_repeated_from_startup, map_folium_2
from functions.utils import get_recommendations
from flask import send_from_directory


with open("./options/cond_env.json", "r", encoding='utf-8') as file:
    options_cond_env = json.load(file)

# Layout de la aplicación
layout = create_layout()

total_data = load_data()  # type: ignore

click_count = 0
div_content = None
data_frame = None
day = 1
map_frame = None
condition = None
count_clicks = -1

click_count_1 = 0

options_dates =  sorted(total_data[0].FECHA.unique())
options_dates = [date.strftime('%Y-%m') for date in options_dates]

options_municipios = sorted(total_data[0].MUN.unique())

options_eventos = ['']
options_tipo_equipo = ['']
options_equipos = ['']
evento_id = None
tipo_equipo_id = None
equipo_id = None

@app.server.route('/graficos_interactivos/<path:filename>', endpoint='serve_graficos_interactivos')
def serve_outputs(filename):
   return send_from_directory("./graficos_interactivos", filename)

layout.children[1]['date-container'].children = dcc.Dropdown(
                id='select-date',
                options=options_dates,
                value=options_dates[0],  # Select the first option as default
                style={'position': 'relative',
                       'width': '100%',
                       'zIndex': 1000,
                       'border': 'none',
                       'color': '#00782b',
                       'font-family': 'DM Sans !important' ,
                       'font-size': '20px'},
                )

layout.children[1]['municipio-container'].children = dcc.Dropdown(
                id='select-municipios',
                options=options_municipios,
                value=options_municipios[0],  # Select the first option as default
                style={'position': 'relative',
                       'width': '100%',
                       'zIndex': 1000,
                       'border': 'none',
                       'color': '#00782b',
                       'font-family': 'DM Sans !important' ,
                       'font-size': '20px'},
                )

layout.children[1]['cond-env-container'].children = dcc.Dropdown(
                id='select-env-condition',
                options=options_cond_env,
                value=options_cond_env[0]['value'],  # Select the first option as default
                style={'position': 'relative',
                       'width': '100%',
                       'zIndex': 1000,
                       'border': 'none',
                       'color': '#00782b',
                       'font-family': 'DM Sans !important' ,
                       'font-size': '20px'},
                )

criticidad_data = {"B38003":{
                    "Tipo_de_equipo":"apoyo",
                    "top_5": {'MATERIAL': "Poste metálico", 'h20-temp': '29.7', 'LONG_APOYO': "15.0", 'TIERRA_PIE': "T", 'h15-wind_gust_spd': "5.0"}},
                    "E31437":{
                    "Tipo_de_equipo":"apoyo",
                    "top_5": {'MATERIAL': "Riel", 'h20-temp': '29.3', 'LONG_APOYO': "9.0", 'TIERRA_PIE': "T", 'h15-wind_gust_spd': "3.0"}},
                    "B38007":{
                    "Tipo_de_equipo": "apoyo",
                    "top_5": {'MATERIAL': "Poste en Concreto", 'h20-temp': '30.1', 'LONG_APOYO': "12.0", 'TIERRA_PIE': "T", 'h15-wind_gust_spd': "1.0"}}}

'''criticidad_data = None'''
'''criticidad_data = {"B38003":{
                    "Tipo_de_equipo":"apoyo",
                    "top_5": {'MATERIAL': "Poste metálico", 'TEMPERATURA': '29.7', 'LONG_APOYO': "15.0", 'TIERRA_PIE': "T", 'VIENTO': "5.0"}},
                    "E31437":{
                    "Tipo_de_equipo":"apoyo",
                    "top_5": {'MATERIAL': "Riel", 'TEMPERATURA': '29.3', 'LONG_APOYO': "9.0", 'TIERRA_PIE': "T", 'VIENTO': "3.0"}},
                    "B38007":{
                    "Tipo_de_equipo": "apoyo",
                    "top_5": {'MATERIAL': "Poste en Concreto", 'TEMPERATURA': '30.1', 'LONG_APOYO': "12.0", 'TIERRA_PIE': "T", 'VIENTO': "1.0"}}}'''

'''criticidad_data = {'Equipo':'Apoyo',
                   'Variable_Valores': {'MATERIAL': 'Torrecilla metalica', 
                                        'LONG_APOYO':'12.0', 
                                        'TIERRA_PIE': 'T', 
                                        'VIENTO': '5.0',
                                        'ÍNDICE_RAYOS': '2.0',
                                        'PRECIPITACIÓN': '4.0',
                                        'RADIACIÓN_UV': '5.0',
                                        'TEMPERATURA': '30.0',
                                        },
                   'Variable_Recomendacion': 'TEMPERATURA'}'''

'''criticidad_data = {'Equipo':'Tramo de red',
                   'Variable_Valores': {'KVNOM': '10.0', 'MATERIALCONDUCTOR':'Cobre', 'CALIBRECONDUCTOR': '12.0', 
                                        'GUARDACONDUCTOR': 'Sí', 'VELOCIDAD_VIENTO': '5.0', 'TEMPERATURA': '20.0',
                                        'HUMEDAD_RELATIVA': '4.0', 'PRECIPITACION': '3.0'},
                   'Variable_Recomendacion': 'CALIBRECONDUCTOR'}'''

'''criticidad_data = {'Equipo':'Transformador',
                   'Variable_Valores': {'KVA': '10.0', 'KV1':'10.2', 'IMPEDANCE': '12.0','LONGITUD': '-75.68', 
                                        'LATITUD': '5.96', 'Temperatura Ambiente': '20.0','Humedad Relativa': '4.0', 'Aceite Aislante': 'Dieléctrico'},
                   'Score_criticidad': {'KVA': 0.25, 'KV1': 0.12, 'Temperatura Ambiente': 0.03, 'Humedad Relativa': 0.32, 'Aceite Aislante': 0.003}}

criticidad_data['Variable_Recomendacion'] = obtener_clave_maximo_score(criticidad_data['Score_criticidad']) '''

'''criticidad_data = {'Equipo':'Switches',
                   'Variable_Valores': {'KV': '10.0', 'PHASES':'3.0', 'STATE': 'Abierto','VELOCIDAD_VIENTO': '4.0', 
                                        'TEMPERATURA_AMBIENTE': '25.96', 'PRECIPITACIÓN': '20.0','HUMEDAD_RELATIVA': '4.0', 'RAYOS': 'No'},
                   'Score_criticidad': {'KV': 0.25, 'PHASES': 0.12, 'TEMPERATURA_AMBIENTE': 0.03, 'HUMEDAD_RELATIVA': 0.32, 'RAYOS': 0.003}}

criticidad_data['Variable_Recomendacion'] = obtener_clave_maximo_score(criticidad_data['Score_criticidad'])'''


# First callback to create the map and slider
@app.callback(
    Output('map-container', 'children'),
    [Input('select-date', 'value'), Input('select-municipios', 'value'), Input("select-env-condition","value"), Input("confirm-button", "n_clicks")],
    State('map-container', 'children')
)
def initialize_map(selected_date, selected_municipios, selected_env_condition, n_clicks, current_map_content):
    global click_count, data_frame, day, condition, options_eventos, dict_options, options_tipo_equipo, options_equipos
    condition = selected_env_condition
    if (n_clicks > click_count) & (condition != 'CRITICIDAD'):
        click_count = n_clicks
        data_frame = select_data(
            int(selected_date[:4]), int(selected_date[5:7]), selected_municipios,
            total_data[0], total_data[1], total_data[2], total_data[3], total_data[4], total_data[5], total_data[6], total_data[8]
        )
        folium_map = map_folium(
            data_frame[0], data_frame[1], data_frame[2], data_frame[3],
            data_frame[4][0], data_frame[5][0], data_frame[6][0],condition
        )

        # Create the map and slider
        map_frame = html.Iframe(
            srcDoc=folium_map,
            style={
                'width': '100%%', 'overflow': 'hidden', 'border': 'none',
                '-ms-overflow-style': 'none', 'scrollbar-width': 'none', 'height': '100%',
                'position': 'relative', 'scrollbar-height': 'none !important', 'margin-bottom': '1vh',
                'object-fit': 'cover', 'overflow-y': 'hidden', 'max-height': '59vh'
            },
            id='folium_map_frame',
            width='100%', height='100%',
        )
        div_content = html.Div([
            html.Div(style={
                'display': 'flex',
                'flexDirection': 'row',
                'alignItems': 'center'},
                children=[
                    dbc.Button(
                            html.I(className="bi bi-arrow-left"),  # Flecha hacia la izquierda
                            id="decrease-btn", color="primary", outline=True, style={
                                'width': '5vh', 
                                'position': 'relative', 
                                'height': '5vh', 
                                'margin': '0 1% 0 0', 
                                'left':'1%',
                                'backgroundImage': "url('/assets/images/left-arrow-direction-svgrepo-com.svg')",
                                'backgroundSize': 'cover',
                                'backgroundPosition': 'center',
                                'backgroundRepeat': 'no-repeat',
                                'border': 'none',
                                'backgroundColor': 'transparent',
                                'cursor': 'pointer'}
                        ),
                    html.Div(style={'width': '102%'}, children=[
                    dcc.Slider(
                        id='date-slider',
                        min=1, max=31, step=1, value=1,
                        tooltip={'always_visible': True, 'placement': 'top'}
                    )]),
                    dbc.Button(
                        html.I(className="bi bi-arrow-right"),  # Flecha hacia la derecha
                        id="increase-btn", color="primary", outline=True, style={
                                'width': '5vh', 
                                'position': 'relative', 
                                'height': '5vh', 
                                'margin': '0 0 0 1%', 
                                'right':'1%',
                                'backgroundImage': "url('/assets/images/left-arrow-direction-svgrepo-com.svg')",
                                'backgroundSize': 'cover',
                                'backgroundPosition': 'center',
                                'backgroundRepeat': 'no-repeat',
                                'transform': 'rotate(180deg)',
                                'border': 'none',
                                'backgroundColor': 'transparent',
                                'cursor': 'pointer'}
                        ),
                ]),
                
            map_frame
            ], style={
                'display': 'flex', 'flexDirection': 'column',
                'height': '100%', 'gap': '10px', 'overflow': 'hidden'})
        return div_content
    
    if (n_clicks > click_count) & (condition == 'CRITICIDAD'):
        click_count = n_clicks
        data_frame = select_data(
            int(selected_date[:4]), int(selected_date[5:7]), selected_municipios,
            total_data[0], total_data[1], total_data[2], total_data[3], total_data[4], total_data[5], total_data[6], total_data[8]
        )
        folium_map = map_folium(
            data_frame[0], data_frame[1], data_frame[2], data_frame[3],
            data_frame[4][0], data_frame[5][0], data_frame[6][0],condition
        )

        options_eventos = list(data_frame[7][day-1]['evento'])
        options_eventos = enumerate_repeated_from_startup(options_eventos)
        index_eventos = list(data_frame[7][day-1]['evento'].index)
        dict_options = dict(zip(options_eventos, index_eventos))
        

        # Create the map and slider
        map_frame = html.Iframe(
            srcDoc=folium_map,
            style={
                'width': '100%', 'overflow': 'hidden', 'border': 'none',
                '-ms-overflow-style': 'none', 'scrollbar-width': 'none', 'height': '100%',
                'position': 'relative', 'scrollbar-height': 'none !important', 'margin-bottom': '1vh',
                'object-fit': 'cover', 'overflow-y': 'hidden', 'max-height': '59vh'
            },
            id='folium_map_frame',
            width='100%', height='100%',
        )
        div_content = html.Div(children = [
            html.Div(style={
                'display': 'flex',
                'flexDirection': 'row',
                'alignItems': 'center',
                'width': '100%'},
                children=[
                    dbc.Button(
                            html.I(className="bi bi-arrow-left"),  # Flecha hacia la izquierda
                            id="decrease-btn", color="primary", outline=True, style={
                                'width': '5vh', 
                                'position': 'relative', 
                                'height': '5vh', 
                                'margin': '0 1% 0 0', 
                                'left':'1%',
                                'backgroundImage': "url('/assets/images/left-arrow-direction-svgrepo-com.svg')",
                                'backgroundSize': 'cover',
                                'backgroundPosition': 'center',
                                'backgroundRepeat': 'no-repeat',
                                'border': 'none',
                                'backgroundColor': 'transparent',
                                'cursor': 'pointer'}
                        ),
                    html.Div(style={'width': '102%'}, children=[
                    dcc.Slider(
                        id='date-slider',
                        min=1, max=31, step=1, value=1,
                        tooltip={'always_visible': True, 'placement': 'top'}
                    )]),
                    dbc.Button(
                        html.I(className="bi bi-arrow-right"),  # Flecha hacia la derecha
                        id="increase-btn", color="primary", outline=True, style={
                                'width': '5vh', 
                                'position': 'relative', 
                                'height': '5vh', 
                                'margin': '0 0 0 1%', 
                                'right':'1%',
                                'backgroundImage': "url('/assets/images/left-arrow-direction-svgrepo-com.svg')",
                                'backgroundSize': 'cover',
                                'backgroundPosition': 'center',
                                'backgroundRepeat': 'no-repeat',
                                'transform': 'rotate(180deg)',
                                'border': 'none',
                                'backgroundColor': 'transparent',
                                'cursor': 'pointer'}
                        ),
                ]),
            html.Div(className='Map_container_2', style={
                'display': 'flex',
                'flexDirection': 'row',
                'alignItems': 'center',
                'height': '51.5vh',
                'width': '100%',  # Add width to ensure full container width
                'margin': '1vh 1vh 1vh 0',
                }, children=[
                    html.Div(className='criterias-container', style={
                        'position': 'relative',
                        'width': '13%',
                        'height': '100%',
                        'backgroundColor': '#16D622',
                        'margin': '0 0 0 1.5%',
                        'borderRadius': '10px',
                        'opacity': '0.7',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'alignItems': 'center',
                        }, children=[
                            html.Div('Evento',className='text-evento', style={
                                'width': '100%',
                                'height': '5%',
                                'lineHeight': '13vh',
                                'color': '#FFFFFF',
                                'fontFamily': "'DM Sans', sans-serif",
                                'fontSize': '25px',
                                'fontWeight': '700',
                                'textAlign': 'center',
                                'display': 'flex',
                                'alignItems': 'center',
                                'justifyContent': 'center',
                                'margin': '1vh 0 0 0',
                                'zIndex': '950',
                            }),
                            html.Div(className='evento-container', style={
                                'width': '70%',
                                'height': '4%',
                                'borderRadius': '5px',
                                'zIndex': '850',
                                'margin': '1vh 0 0 0',
                            }, children=[
                                dcc.Dropdown(
                                    id='select-evento',
                                    options=options_eventos,
                                    value='',  # Select the first option as default
                                style={'position': 'relative',
                                    'width': '100%',
                                    'height': '4%',
                                    'border': 'none',
                                    'color': '#00782b',
                                    'font-family': 'DM Sans !important' ,
                                    'font-size': '20px',
                                    'zIndex': '850',},
                                )
                            ]),
                            html.Button('Equipos críticos', id='equipos-criticos-button',className='equipos-criticos-button', style={
                                        'fontFamily': "'DM Sans', sans-serif",
                                        'fontSize': '16px',
                                        'fontWeight': '700',
                                        'color': 'black',
                                        'cursor': 'pointer',
                                        'borderRadius': '3px',
                                        'borderColor': 'white',
                                        'width': '82%',
                                        'height': '8%',
                                        'backgroundColor': '#11BB52CF',
                                        'position': 'relative',
                                        'zIndex': '350',
                                        'margin': '4vh 0 0 0',
                                    }, n_clicks=0),
                            html.Button('Recomendación', id='recomendacion-button',className='recomendacion-button', style={
                                        'fontFamily': "'DM Sans', sans-serif",
                                        'fontSize': '16px',
                                        'fontWeight': '700',
                                        'color': 'black',
                                        'cursor': 'pointer',
                                        'borderRadius': '3px',
                                        'borderColor': 'white',
                                        'width': '82%',
                                        'height': '8%',
                                        'backgroundColor': '#11BB52CF',
                                        'position': 'relative',
                                        'zIndex': '350',
                                        'margin': '3vh 0 0 0',
                                    }, n_clicks=0),
                        ]),
                    html.Div(className='map-map-2', style={
                        'position': 'relative', 
                        'width': '85%',  # Adjust width to fit within the container
                        'height': '100%'  # Ensure full height
                    }, children=[map_frame])
                ])
                ], style={
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'center',
                })
        return div_content
    return dash.no_update

@app.callback(
        Output('recomendacion-button', 'children'),
        Input('select-evento', 'value'))
def update_evento(evento_value):
    global evento_id
    if evento_value != evento_id:
        evento_id = evento_value
    return dash.no_update


@app.callback(
    [Output('date-slider', 'value'), 
     Output('folium_map_frame', 'srcDoc'), 
     Output('select-evento', 'options')],
    [Input('decrease-btn', 'n_clicks'),
     Input('increase-btn', 'n_clicks'),
     Input('date-slider', 'value'),
     Input('equipos-criticos-button', 'n_clicks')],
    [State('date-slider', 'min'),
     State('date-slider', 'max')]
)
def update_maps_and_slider(decrease_clicks, increase_clicks, slider_value, equipos_clicks, slider_min, slider_max):
    global day, data_frame, condition, options_eventos, dict_options, click_count_1, evento_id
    
    triggered_id = ctx.triggered_id
    
    if triggered_id == "equipos-criticos-button":
        if (equipos_clicks > click_count_1) and (condition == 'CRITICIDAD') and (evento_id is not None and evento_id != ''):
            click_count_1 = equipos_clicks
            map = map_folium_2(data_frame[0], data_frame[1], data_frame[2], data_frame[3],
                data_frame[4][day-1], dict_options[evento_id], total_data[7], total_data[8], 
                total_data[9], total_data[10], total_data[11],total_data[12],total_data[13])
            return dash.no_update, map, dash.no_update
    
    decrease_clicks = decrease_clicks or 0
    increase_clicks = increase_clicks or 0

    if triggered_id == "decrease-btn" and slider_value > slider_min:
        slider_value -= 1
    elif triggered_id == "increase-btn" and slider_value < slider_max:
        slider_value += 1
    
    if slider_value == day:
        return slider_value, dash.no_update, dash.no_update

    day = slider_value

    if condition == 'CRITICIDAD':
        folium_map = map_folium(data_frame[0], data_frame[1], data_frame[2], data_frame[3],
            data_frame[4][day-1], data_frame[5][day-1], data_frame[6][day-1], condition)
        options_eventos = list(data_frame[7][day-1]['evento'])
        options_eventos = enumerate_repeated_from_startup(options_eventos)
        index_eventos = list(data_frame[7][day-1]['evento'].index)
        dict_options = dict(zip(options_eventos, index_eventos))
        return slider_value, folium_map, options_eventos
    else:
        folium_map = map_folium(data_frame[0], data_frame[1], data_frame[2], data_frame[3],
            data_frame[4][day-1], data_frame[5][day-1], data_frame[6][day-1], condition)
        return slider_value, folium_map, dash.no_update

@app.callback(
    Output('url-maps', 'pathname'),
    [
        Input('recomendacion-button', 'n_clicks'),
        Input('button-chat', 'n_clicks'),
        Input('button-graphs', 'n_clicks'),
        Input('button-tab-net', 'n_clicks')
    ],
    prevent_initial_call=True
)
def handle_url_update(select_evento_clicks, n_clicks_chat, n_clicks_graphs, n_clicks_tab_net):
    global evento_id

    # Obtener el contexto del trigger
    ctx = callback_context
    if not ctx.triggered:
        raise exceptions.PreventUpdate

    # Identificar qué entrada disparó el callback
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Lógica para 'select-evento'
    if triggered_id == 'recomendacion-button' and select_evento_clicks:
        if (select_evento_clicks != evento_id) and (select_evento_clicks != None):
            evento_id = select_evento_clicks
            get_recommendations(criticidad_data)
            return "/chat_page"  # Cambia esto al pathname correspondiente
        

    # Lógica para los botones
    if triggered_id == 'button-chat' and n_clicks_chat:
        return "/chat_page"
    elif triggered_id == 'button-graphs' and n_clicks_graphs:
        return "/graphs_page"
    elif triggered_id == 'button-tab-net' and n_clicks_tab_net:
        return "/tab-net_page"

    raise exceptions.PreventUpdate

 