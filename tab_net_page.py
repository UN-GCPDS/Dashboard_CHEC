import dash
from dash import Dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Output, Input
from dash import callback_context, exceptions
from dash.dependencies import Input, Output, State

import os
import shutil
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
import json
from flask import send_from_directory
from ui_components.ui_tab_net import create_layout
from utils.tab_net_functions import load_data, graphics_PDF

data_total = load_data()

layout = create_layout()

selection_criteria = [['','','',''],['','','',''],['','','',''],['','','','']]
count_clicks = -1
count = 0
with open('count_tab_net.txt', 'w', encoding='utf-8') as file:
    file.write(str(count))

# Para la carpeta outputs_PDFs 
@app.server.route('/outputs_PDFs/<path:filename>', endpoint='serve_pdf_outputs')
def serve_pdf_outputs(filename):
   return send_from_directory("./outputs_PDFs", filename)

            
@app.callback(
    Output('sub-criteria-1-filters-container-tab-net', 'children'),
    Input('select-subcriteria-1-tab-net', 'value'),
)

def select_subcriteria_1(selection_sub_criteria_1):

        global selection_criteria

        if selection_sub_criteria_1 != '':

                match data_total[6][selection_sub_criteria_1].dtype:

                        case 'O' | 'int64' | 'int32':

                                selection_criteria[0][0] = 'seleccion'
                                selection_criteria[0][1] = selection_sub_criteria_1
                                select_subcriteria = ['']
                                select_subcriteria.extend(data_total[6][selection_sub_criteria_1].unique())

                                div = [ 
                                        html.Div('Selección:',className='sub-criteria-1-1-filter-text-tab-net',
                                        style={
                                                'width': '20%',
                                                'height': '100%',
                                                'color': '#FFFFFF',
                                                'fontFamily': "'DM Sans', sans-serif",
                                                'fontSize': '130%',
                                                'fontWeight': '700',
                                                'textAlign': 'center',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center',
                                        }),
                                        html.Div(className='sub-criteria-1-1-filter-container-tab-net',
                                        style={
                                                'width': '50%',
                                                'height': '100%',
                                                'borderRadius': '5px',
                                                'backgroundColor': 'white',
                                                'margin': '0 0 0 9%'
                                        }, children = [
                                        dcc.Dropdown(
                                                id='select-subcriteria-1-1-tab-net',
                                                options=select_subcriteria,
                                                value=select_subcriteria[0],  # Select the first option as default
                                                style={'position': 'relative',
                                                        'width': '100%',
                                                        'zIndex': 925,
                                                        'border': 'none',
                                                        'color': '#00782b',
                                                        'font-family': 'DM Sans !important' ,
                                                        'font-size': '20px'},
                                                )
                                        ])
                                ]

                                return div
                
                        case 'float32' | 'float64' | 'float16':
                        
                                selection_criteria[0][0] = 'rango_num'
                                selection_criteria[0][1] = selection_sub_criteria_1

                                div = [
                                        html.Div('Operador:',className='sub-criteria-1-1-filter-text-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'color': '#FFFFFF',
                                                'fontFamily': "'DM Sans', sans-serif",
                                                'fontSize': '131%',
                                                'fontWeight': '700',
                                                'textAlign': 'center',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center',
                                                'margin': '0 0 0 2%'
                                                }),
                                        html.Div(className='sub-criteria-1-1-filter-container-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'borderRadius': '5px',
                                                'backgroundColor': 'white',
                                                'margin': '0 0 0 8%'
                                                }, children=[
                                                dcc.Dropdown(
                                                                id='select-subcriteria-1-1-tab-net',
                                                                options=['', '>', '>=','<','<=','!=','=='],
                                                                value='',  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 925,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '20px'},
                                                                )
                                                ]),
                                        html.Div('Valor:',className='sub-criteria-1-2-filter-text-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'color': '#FFFFFF',
                                                'fontFamily': "'DM Sans', sans-serif",
                                                'fontSize': '130%',
                                                'fontWeight': '700',
                                                'textAlign': 'center',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center',
                                                'margin': '0 0 0 0'
                                                }),
                                        html.Div(className='sub-criteria-1-2-filter-container-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'borderRadius': '5px',
                                                'backgroundColor': 'white',
                                                'margin': '0 0 0 0%',
                                                }, children = [
                                                dcc.Input(
                                                        id='select-subcriteria-1-2-tab-net',
                                                        type='number',
                                                        placeholder='Ingresa un valor',
                                                        style={'position': 'relative',
                                                                'width': '91.0%',
                                                                'zIndex': 925,
                                                                'border': 'none',
                                                                'color': '#00782b',
                                                                'font-family': 'DM Sans !important' ,
                                                                'font-size': '20px',
                                                                'height': '77%',
                                                                'transform': 'translate(1%, 11%)'
                                                                },
                                                        )]
                                                        
                                                ),    
                                ]
                                
                                return div
                        
                        case 'datetime64[ns]' | 'period[M]':

                                selection_criteria[0][0] = 'fecha'
                                selection_criteria[0][1] = selection_sub_criteria_1
                                options_dates =  sorted(data_total[6][selection_sub_criteria_1].unique())
                                options_dates = [date.strftime('%Y-%m-%d') for date in options_dates]
                                options_dates = list(sorted(set(options_dates)))
                        
                                div = [ html.Div('Desde:',className='sub-criteria-1-1-filter-text-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'color': '#FFFFFF',
                                                'fontFamily': "'DM Sans', sans-serif",
                                                'fontSize': '130%',
                                                'fontWeight': '700',
                                                'textAlign': 'center',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center',
                                                }),
                                        html.Div(className='sub-criteria-1-1-filter-container-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'borderRadius': '5px',
                                                'backgroundColor': 'white',
                                                'margin': '0 0 0 3%'
                                                }, children = [
                                                        dcc.Dropdown(
                                                                id='select-subcriteria-1-1-tab-net',
                                                                options=options_dates,
                                                                value='',  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 925,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '15px'},
                                                        )
                                                ]),
                                        html.Div('Hasta:',className='sub-criteria-1-2-filter-text-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'color': '#FFFFFF',
                                                'fontFamily': "'DM Sans', sans-serif",
                                                'fontSize': '130%',
                                                'fontWeight': '700',
                                                'textAlign': 'center',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center',
                                                'margin': '0 0 0 3%'
                                                }),
                                        html.Div(className='sub-criteria-1-2-filter-container-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'borderRadius': '5px',
                                                'backgroundColor': 'white',
                                                'margin': '0 0 0 3%'
                                                }, children=[
                                                        dcc.Dropdown(
                                                                id='select-subcriteria-1-2-tab-net',
                                                                options=options_dates,
                                                                value='',  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 925,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '15px'},
                                                        )
                                                ]),    
                                ]
                                
                                return div

                        case _:
                                selection_criteria[0][0] = ''
                                return dash.no_update

        else:
                selection_criteria[0][0] = ''
                return None

@app.callback(
    Input('select-subcriteria-1-1-tab-net', 'value'),
)
def select_subcriteria_1_1(selection_sub_criteria_1_1):
        global selection_criteria
        selection_criteria[0][2] = selection_sub_criteria_1_1

@app.callback(
    Input('select-subcriteria-1-2-tab-net', 'value'),
)
def select_subcriteria_1_2(selection_sub_criteria_1_2):
        global selection_criteria
        selection_criteria[0][3] = selection_sub_criteria_1_2



@app.callback(
    Output('sub-criteria-2-filters-container-tab-net', 'children'),
    Input('select-subcriteria-2-tab-net', 'value'),
)

def select_subcriteria_2(selection_sub_criteria_2):

        global selection_criteria

        if selection_sub_criteria_2 != '':

                match data_total[6][selection_sub_criteria_2].dtype:

                        case 'O' | 'int64' | 'int32':

                                selection_criteria[1][0] = 'seleccion'
                                selection_criteria[1][1] = selection_sub_criteria_2
                                select_subcriteria = ['']
                                select_subcriteria.extend(data_total[6][selection_sub_criteria_2].unique())

                                div = [ 
                                        html.Div('Selección:',className='sub-criteria-2-1-filter-text-tab-net',
                                        style={
                                                'width': '20%',
                                                'height': '100%',
                                                'color': '#FFFFFF',
                                                'fontFamily': "'DM Sans', sans-serif",
                                                'fontSize': '130%',
                                                'fontWeight': '700',
                                                'textAlign': 'center',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center',
                                        }),
                                        html.Div(className='sub-criteria-2-1-filter-container-tab-net',
                                        style={
                                                'width': '50%',
                                                'height': '100%',
                                                'borderRadius': '5px',
                                                'backgroundColor': 'white',
                                                'margin': '0 0 0 9%'
                                        }, children = [
                                        dcc.Dropdown(
                                                id='select-subcriteria-2-1-tab-net',
                                                options=select_subcriteria,
                                                value=select_subcriteria[0],  # Select the first option as default
                                                style={'position': 'relative',
                                                        'width': '100%',
                                                        'zIndex': 825,
                                                        'border': 'none',
                                                        'color': '#00782b',
                                                        'font-family': 'DM Sans !important' ,
                                                        'font-size': '20px'},
                                                )
                                        ])
                                ]

                                return div
                
                        case 'float32' | 'float64' | 'float16':
                        
                                selection_criteria[1][0] = 'rango_num'
                                selection_criteria[1][1] = selection_sub_criteria_2

                                div = [
                                        html.Div('Operador:',className='sub-criteria-2-1-filter-text-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'color': '#FFFFFF',
                                                'fontFamily': "'DM Sans', sans-serif",
                                                'fontSize': '131%',
                                                'fontWeight': '700',
                                                'textAlign': 'center',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center',
                                                'margin': '0 0 0 2%'
                                                }),
                                        html.Div(className='sub-criteria-2-1-filter-container-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'borderRadius': '5px',
                                                'backgroundColor': 'white',
                                                'margin': '0 0 0 8%'
                                                }, children=[
                                                dcc.Dropdown(
                                                                id='select-subcriteria-2-1-tab-net',
                                                                options=['', '>', '>=','<','<=','!=','=='],
                                                                value='',  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 825,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '20px'},
                                                                )
                                                ]),
                                        html.Div('Valor:',className='sub-criteria-2-2-filter-text-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'color': '#FFFFFF',
                                                'fontFamily': "'DM Sans', sans-serif",
                                                'fontSize': '130%',
                                                'fontWeight': '700',
                                                'textAlign': 'center',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center',
                                                'margin': '0 0 0 0'
                                                }),
                                        html.Div(className='sub-criteria-2-2-filter-container-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'borderRadius': '5px',
                                                'backgroundColor': 'white',
                                                'margin': '0 0 0 0%',
                                                }, children = [
                                                dcc.Input(
                                                        id='select-subcriteria-2-2-tab-net',
                                                        type='number',
                                                        placeholder='Ingresa un valor',
                                                        style={'position': 'relative',
                                                                'width': '91.0%',
                                                                'zIndex': 825,
                                                                'border': 'none',
                                                                'color': '#00782b',
                                                                'font-family': 'DM Sans !important' ,
                                                                'font-size': '20px',
                                                                'height': '77%',
                                                                'transform': 'translate(1%, 11%)'
                                                                },
                                                        )]
                                                        
                                                ),    
                                ]
                                
                                return div
                        
                        case 'datetime64[ns]' | 'period[M]':

                                selection_criteria[1][0] = 'fecha'
                                selection_criteria[1][1] = selection_sub_criteria_2
                                options_dates =  sorted(data_total[6][selection_sub_criteria_2].unique())
                                options_dates = [date.strftime('%Y-%m-%d') for date in options_dates]
                                options_dates = list(sorted(set(options_dates)))
                        
                                div = [ html.Div('Desde:',className='sub-criteria-2-1-filter-text-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'color': '#FFFFFF',
                                                'fontFamily': "'DM Sans', sans-serif",
                                                'fontSize': '130%',
                                                'fontWeight': '700',
                                                'textAlign': 'center',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center',
                                                }),
                                        html.Div(className='sub-criteria-2-1-filter-container-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'borderRadius': '5px',
                                                'backgroundColor': 'white',
                                                'margin': '0 0 0 3%'
                                                }, children = [
                                                        dcc.Dropdown(
                                                                id='select-subcriteria-2-1-tab-net',
                                                                options=options_dates,
                                                                value='',  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 825,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '15px'},
                                                        )
                                                ]),
                                        html.Div('Hasta:',className='sub-criteria-2-2-filter-text-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'color': '#FFFFFF',
                                                'fontFamily': "'DM Sans', sans-serif",
                                                'fontSize': '130%',
                                                'fontWeight': '700',
                                                'textAlign': 'center',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center',
                                                'margin': '0 0 0 3%'
                                                }),
                                        html.Div(className='sub-criteria-2-2-filter-container-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'borderRadius': '5px',
                                                'backgroundColor': 'white',
                                                'margin': '0 0 0 3%'
                                                }, children=[
                                                        dcc.Dropdown(
                                                                id='select-subcriteria-2-2-tab-net',
                                                                options=options_dates,
                                                                value='',  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 825,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '15px'},
                                                        )
                                                ]),    
                                ]
                                
                                return div

                        case _:
                                selection_criteria[1][0] = ''
                                return dash.no_update

        else:
                selection_criteria[1][0] = ''
                return None

@app.callback(
    Input('select-subcriteria-2-1-tab-net', 'value'),
)
def select_subcriteria_2_1(selection_sub_criteria_2_1):
        global selection_criteria
        selection_criteria[1][2] = selection_sub_criteria_2_1

@app.callback(
    Input('select-subcriteria-2-2-tab-net', 'value'),
)
def select_subcriteria_2_2(selection_sub_criteria_2_2):
        global selection_criteria
        selection_criteria[1][3] = selection_sub_criteria_2_2



@app.callback(
    Output('sub-criteria-3-filters-container-tab-net', 'children'),
    Input('select-subcriteria-3-tab-net', 'value'),
)

def select_subcriteria_3(selection_sub_criteria_3):

        global selection_criteria

        if selection_sub_criteria_3 != '':

                match data_total[6][selection_sub_criteria_3].dtype:

                        case 'O' | 'int64' | 'int32':

                                selection_criteria[2][0] = 'seleccion'
                                selection_criteria[2][1] = selection_sub_criteria_3
                                select_subcriteria = ['']
                                select_subcriteria.extend(data_total[6][selection_sub_criteria_3].unique())

                                div = [ 
                                        html.Div('Selección:',className='sub-criteria-3-1-filter-text-tab-net',
                                        style={
                                                'width': '20%',
                                                'height': '100%',
                                                'color': '#FFFFFF',
                                                'fontFamily': "'DM Sans', sans-serif",
                                                'fontSize': '130%',
                                                'fontWeight': '700',
                                                'textAlign': 'center',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center',
                                        }),
                                        html.Div(className='sub-criteria-3-1-filter-container-tab-net',
                                        style={
                                                'width': '50%',
                                                'height': '100%',
                                                'borderRadius': '5px',
                                                'backgroundColor': 'white',
                                                'margin': '0 0 0 9%'
                                        }, children = [
                                        dcc.Dropdown(
                                                id='select-subcriteria-3-1-tab-net',
                                                options=select_subcriteria,
                                                value=select_subcriteria[0],  # Select the first option as default
                                                style={'position': 'relative',
                                                        'width': '100%',
                                                        'zIndex': 725,
                                                        'border': 'none',
                                                        'color': '#00782b',
                                                        'font-family': 'DM Sans !important' ,
                                                        'font-size': '20px'},
                                                )
                                        ])
                                ]

                                return div
                
                        case 'float32' | 'float64' | 'float16':
                        
                                selection_criteria[2][0] = 'rango_num'
                                selection_criteria[2][1] = selection_sub_criteria_3

                                div = [
                                        html.Div('Operador:',className='sub-criteria-3-1-filter-text-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'color': '#FFFFFF',
                                                'fontFamily': "'DM Sans', sans-serif",
                                                'fontSize': '131%',
                                                'fontWeight': '700',
                                                'textAlign': 'center',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center',
                                                'margin': '0 0 0 2%'
                                                }),
                                        html.Div(className='sub-criteria-3-1-filter-container-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'borderRadius': '5px',
                                                'backgroundColor': 'white',
                                                'margin': '0 0 0 8%'
                                                }, children=[
                                                dcc.Dropdown(
                                                                id='select-subcriteria-3-1-tab-net',
                                                                options=['', '>', '>=','<','<=','!=','=='],
                                                                value='',  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 725,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '20px'},
                                                                )
                                                ]),
                                        html.Div('Valor:',className='sub-criteria-3-2-filter-text-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'color': '#FFFFFF',
                                                'fontFamily': "'DM Sans', sans-serif",
                                                'fontSize': '130%',
                                                'fontWeight': '700',
                                                'textAlign': 'center',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center',
                                                'margin': '0 0 0 0'
                                                }),
                                        html.Div(className='sub-criteria-3-2-filter-container-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'borderRadius': '5px',
                                                'backgroundColor': 'white',
                                                'margin': '0 0 0 0%',
                                                }, children = [
                                                dcc.Input(
                                                        id='select-subcriteria-3-2-tab-net',
                                                        type='number',
                                                        placeholder='Ingresa un valor',
                                                        style={'position': 'relative',
                                                                'width': '91.0%',
                                                                'zIndex': 725,
                                                                'border': 'none',
                                                                'color': '#00782b',
                                                                'font-family': 'DM Sans !important' ,
                                                                'font-size': '20px',
                                                                'height': '77%',
                                                                'transform': 'translate(1%, 11%)'
                                                                },
                                                        )]
                                                        
                                                ),    
                                ]
                                
                                return div
                        
                        case 'datetime64[ns]' | 'period[M]':

                                selection_criteria[2][0] = 'fecha'
                                selection_criteria[2][1] = selection_sub_criteria_3
                                options_dates =  sorted(data_total[6][selection_sub_criteria_3].unique())
                                options_dates = [date.strftime('%Y-%m-%d') for date in options_dates]
                                options_dates = list(sorted(set(options_dates)))
                        
                                div = [ html.Div('Desde:',className='sub-criteria-3-1-filter-text-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'color': '#FFFFFF',
                                                'fontFamily': "'DM Sans', sans-serif",
                                                'fontSize': '130%',
                                                'fontWeight': '700',
                                                'textAlign': 'center',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center',
                                                }),
                                        html.Div(className='sub-criteria-3-1-filter-container-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'borderRadius': '5px',
                                                'backgroundColor': 'white',
                                                'margin': '0 0 0 3%'
                                                }, children = [
                                                        dcc.Dropdown(
                                                                id='select-subcriteria-3-1-tab-net',
                                                                options=options_dates,
                                                                value='',  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 725,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '15px'},
                                                        )
                                                ]),
                                        html.Div('Hasta:',className='sub-criteria-3-2-filter-text-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'color': '#FFFFFF',
                                                'fontFamily': "'DM Sans', sans-serif",
                                                'fontSize': '130%',
                                                'fontWeight': '700',
                                                'textAlign': 'center',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center',
                                                'margin': '0 0 0 3%'
                                                }),
                                        html.Div(className='sub-criteria-3-2-filter-container-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'borderRadius': '5px',
                                                'backgroundColor': 'white',
                                                'margin': '0 0 0 3%'
                                                }, children=[
                                                        dcc.Dropdown(
                                                                id='select-subcriteria-3-2-tab-net',
                                                                options=options_dates,
                                                                value='',  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 725,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '15px'},
                                                        )
                                                ]),    
                                ]
                                
                                return div

                        case _:
                                selection_criteria[2][0] = ''
                                return dash.no_update

        else:
                selection_criteria[2][0] = ''
                return None

@app.callback(
    Input('select-subcriteria-3-1-tab-net', 'value'),
)
def select_subcriteria_3_1(selection_sub_criteria_3_1):
        global selection_criteria
        selection_criteria[2][2] = selection_sub_criteria_3_1

@app.callback(
    Input('select-subcriteria-3-2-tab-net', 'value'),
)
def select_subcriteria_3_2(selection_sub_criteria_3_2):
        global selection_criteria
        selection_criteria[2][3] = selection_sub_criteria_3_2



@app.callback(
    Output('sub-criteria-4-filters-container-tab-net', 'children'),
    Input('select-subcriteria-4-tab-net', 'value'),
)

def select_subcriteria_4(selection_sub_criteria_4):

        global selection_criteria

        if selection_sub_criteria_4 != '':

                match data_total[6][selection_sub_criteria_4].dtype:

                        case 'O' | 'int64' | 'int32':

                                selection_criteria[3][0] = 'seleccion'
                                selection_criteria[3][1] = selection_sub_criteria_4
                                select_subcriteria = ['']
                                select_subcriteria.extend(data_total[6][selection_sub_criteria_4].unique())

                                div = [ 
                                        html.Div('Selección:',className='sub-criteria-4-1-filter-text-tab-net',
                                        style={
                                                'width': '20%',
                                                'height': '100%',
                                                'color': '#FFFFFF',
                                                'fontFamily': "'DM Sans', sans-serif",
                                                'fontSize': '130%',
                                                'fontWeight': '700',
                                                'textAlign': 'center',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center',
                                        }),
                                        html.Div(className='sub-criteria-4-1-filter-container-tab-net',
                                        style={
                                                'width': '50%',
                                                'height': '100%',
                                                'borderRadius': '5px',
                                                'backgroundColor': 'white',
                                                'margin': '0 0 0 9%'
                                        }, children = [
                                        dcc.Dropdown(
                                                id='select-subcriteria-4-1-tab-net',
                                                options=select_subcriteria,
                                                value=select_subcriteria[0],  # Select the first option as default
                                                style={'position': 'relative',
                                                        'width': '100%',
                                                        'zIndex': 625,
                                                        'border': 'none',
                                                        'color': '#00782b',
                                                        'font-family': 'DM Sans !important' ,
                                                        'font-size': '20px'},
                                                )
                                        ])
                                ]

                                return div
                
                        case 'float32' | 'float64' | 'float16':
                        
                                selection_criteria[3][0] = 'rango_num'
                                selection_criteria[3][1] = selection_sub_criteria_4

                                div = [
                                        html.Div('Operador:',className='sub-criteria-4-1-filter-text-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'color': '#FFFFFF',
                                                'fontFamily': "'DM Sans', sans-serif",
                                                'fontSize': '131%',
                                                'fontWeight': '700',
                                                'textAlign': 'center',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center',
                                                'margin': '0 0 0 2%'
                                                }),
                                        html.Div(className='sub-criteria-4-1-filter-container-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'borderRadius': '5px',
                                                'backgroundColor': 'white',
                                                'margin': '0 0 0 8%'
                                                }, children=[
                                                dcc.Dropdown(
                                                                id='select-subcriteria-4-1-tab-net',
                                                                options=['', '>', '>=','<','<=','!=','=='],
                                                                value='',  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 625,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '20px'},
                                                                )
                                                ]),
                                        html.Div('Valor:',className='sub-criteria-4-2-filter-text-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'color': '#FFFFFF',
                                                'fontFamily': "'DM Sans', sans-serif",
                                                'fontSize': '130%',
                                                'fontWeight': '700',
                                                'textAlign': 'center',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center',
                                                'margin': '0 0 0 0'
                                                }),
                                        html.Div(className='sub-criteria-4-2-filter-container-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'borderRadius': '5px',
                                                'backgroundColor': 'white',
                                                'margin': '0 0 0 0%',
                                                }, children = [
                                                dcc.Input(
                                                        id='select-subcriteria-4-2-tab-net',
                                                        type='number',
                                                        placeholder='Ingresa un valor',
                                                        style={'position': 'relative',
                                                                'width': '91.0%',
                                                                'zIndex': 625,
                                                                'border': 'none',
                                                                'color': '#00782b',
                                                                'font-family': 'DM Sans !important' ,
                                                                'font-size': '20px',
                                                                'height': '77%',
                                                                'transform': 'translate(1%, 11%)'
                                                                },
                                                        )]
                                                        
                                                ),    
                                ]
                                
                                return div
                        
                        case 'datetime64[ns]' | 'period[M]':

                                selection_criteria[3][0] = 'fecha'
                                selection_criteria[3][1] = selection_sub_criteria_4
                                options_dates =  sorted(data_total[6][selection_sub_criteria_4].unique())
                                options_dates = [date.strftime('%Y-%m-%d') for date in options_dates]
                                options_dates = list(sorted(set(options_dates)))
                        
                                div = [ html.Div('Desde:',className='sub-criteria-4-1-filter-text-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'color': '#FFFFFF',
                                                'fontFamily': "'DM Sans', sans-serif",
                                                'fontSize': '130%',
                                                'fontWeight': '700',
                                                'textAlign': 'center',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center',
                                                }),
                                        html.Div(className='sub-criteria-4-1-filter-container-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'borderRadius': '5px',
                                                'backgroundColor': 'white',
                                                'margin': '0 0 0 3%'
                                                }, children = [
                                                        dcc.Dropdown(
                                                                id='select-subcriteria-4-1-tab-net',
                                                                options=options_dates,
                                                                value='',  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 625,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '15px'},
                                                        )
                                                ]),
                                        html.Div('Hasta:',className='sub-criteria-4-2-filter-text-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'color': '#FFFFFF',
                                                'fontFamily': "'DM Sans', sans-serif",
                                                'fontSize': '130%',
                                                'fontWeight': '700',
                                                'textAlign': 'center',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center',
                                                'margin': '0 0 0 3%'
                                                }),
                                        html.Div(className='sub-criteria-4-2-filter-container-tab-net',
                                                style={
                                                'width': '20%',
                                                'height': '100%',
                                                'borderRadius': '5px',
                                                'backgroundColor': 'white',
                                                'margin': '0 0 0 3%'
                                                }, children=[
                                                        dcc.Dropdown(
                                                                id='select-subcriteria-4-2-tab-net',
                                                                options=options_dates,
                                                                value='',  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 625,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '15px'},
                                                        )
                                                ]),    
                                ]
                                
                                return div

                        case _:
                                selection_criteria[3][0] = ''
                                return dash.no_update

        else:
                selection_criteria[3][0] = ''
                return None

@app.callback(
    Input('select-subcriteria-4-1-tab-net', 'value'),
)
def select_subcriteria_4_1(selection_sub_criteria_4_1):
        global selection_criteria
        selection_criteria[3][2] = selection_sub_criteria_4_1

@app.callback(
    Input('select-subcriteria-4-2-tab-net', 'value'),
)
def select_subcriteria_4_2(selection_sub_criteria_4_2):
        global selection_criteria
        selection_criteria[3][3] = selection_sub_criteria_4_2

                
@app.callback(
        Output('graph-container', 'children'),
        Input('confirm-button-ok-tab-net', 'n_clicks')   
)
def confirm_button_fn_tab_net(n_clicks):
    global count_clicks
    global count
    
    # Si no se ha hecho clic aún (n_clicks es None)
    if n_clicks is None:
        return dash.no_update
    
    # Si no ha habido un cambio de clics
    if n_clicks <= count_clicks:
        return dash.no_update

    # Actualiza el contador de clics
    count_clicks = n_clicks
    
    # Verificar condiciones para la generación del texto
    if (selection_criteria[0][0] != '') | (selection_criteria[1][0] != '') | (selection_criteria[2][0] != '') | (selection_criteria[3][0] != ''):

        with open('count_tab_net.txt', 'r', encoding='utf-8') as file:
                count = file.read()
        count = int(count)
        if count > 0:
               
                # Especifica la ruta de la carpeta
                carpeta = "./outputs_PDFs"

                # Verifica si la carpeta existe
                if os.path.exists(carpeta):
                        # Itera por todos los archivos en la carpeta
                        for archivo in os.listdir(carpeta):
                                # Comprueba si el archivo tiene la extensión .png
                                if archivo.endswith(".png"):
                                        ruta_archivo = os.path.join(carpeta, archivo)
                                        # Elimina el archivo
                                        os.remove(ruta_archivo)
                                        print(f"Eliminado: {ruta_archivo}")
                        print("Todos los archivos .png han sido eliminados.")

        # Llamada a la función de generación de gráfico
        graphics_PDF(selection_criteria, data_total[6], data_total[0], data_total[5], data_total[7], data_total[8], data_total[9], count)

        with open('count_tab_net.txt', 'r', encoding='utf-8') as file:
                count = file.read()
        count = int(count)
        # Estilo del gráfico
        graphics = html.Iframe(
                src="/outputs_PDFs/graphics_criticality_"+str(count)+".pdf",  # Ruta relativa al archivo servido
                style={"width": "95%", "height": "95%","position": "relative", "margin": "2% 0 0 0"},  # Ajusta las dimensiones
                )
        
        
        count = count + 1
        with open('count_graphs.txt', 'w', encoding='utf-8') as file:
                file.write(str(count))

        return graphics

    # Si no se cumplen las condiciones
    return dash.no_update

@app.callback(
    Output('url-tab-net', 'pathname'),
    [Input('button-chat', 'n_clicks'),
     Input('button-maps', 'n_clicks'),
     Input('button-graphs', 'n_clicks')]
)
def redirect_to_pages(n_clicks_chat, n_clicks_maps, n_clicks_graphs):
    # Obtener el contexto del trigger
    ctx = callback_context
    if not ctx.triggered:
        raise exceptions.PreventUpdate

    # Identificar qué botón disparó el callback
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'button-chat' and n_clicks_chat:
        return "/chat_page"
    elif triggered_id == 'button-maps' and n_clicks_maps:
        return "/maps_page"
    elif triggered_id == 'button-graphs' and n_clicks_graphs:
        return "/graphs_page"

    raise exceptions.PreventUpdate