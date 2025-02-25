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
from ui_components.ui_graphs import create_layout
from utils.graphs_functions import load_data, graph_probabilty


layout = create_layout()

data_total = load_data()

selection_criteria = [[''],['','','',''],['','','',''],['','','',''],['','','',''],['']]
count_clicks = -1
count = 0

# Para la carpeta outputs
@app.server.route('/outputs/<path:filename>', endpoint='serve_outputs')
def serve_outputs(filename):
   return send_from_directory("./outputs", filename)

# First callback to create the map and slider
@app.callback(
    [Output('sub-criteria-1-container', 'children'),Output('sub-criteria-2-container', 'children'),Output('sub-criteria-3-container', 'children'), Output('sub-criteria-4-container', 'children'),Output('target-variable-container', 'children')],
    Input('select-criteria', 'value'),
)
def select_main_criteria(select_criteria):
    global selection_criteria

    match select_criteria:

        case 'Eventos Interruptor':
              selection_criteria[0] = select_criteria
              select_subcriteria_1 = ['']
              select_subcriteria_1.extend(data_total[0].columns.to_list())
              dropdown_subcriteria_1 = dcc.Dropdown(
                  id='select-subcriteria-1',
                  options=select_subcriteria_1,
                  value=select_subcriteria_1[0],  # Select the first option as default
                  style={'position': 'relative',
                        'width': '100%',
                        'zIndex': 900,
                        'border': 'none',
                        'color': '#00782b',
                        'font-family': 'DM Sans !important' ,
                        'font-size': '20px'},
                )
              
              select_subcriteria_2 = ['']
              select_subcriteria_2.extend(data_total[0].columns.to_list())
              dropdown_subcriteria_2 = dcc.Dropdown(
                  id='select-subcriteria-2',
                  options=select_subcriteria_2,
                  value=select_subcriteria_2[0],  # Select the first option as default
                  style={'position': 'relative',
                        'width': '100%',
                        'zIndex': 800,
                        'border': 'none',
                        'color': '#00782b',
                        'font-family': 'DM Sans !important' ,
                        'font-size': '20px'},
                )
              
              select_subcriteria_3 = ['']
              select_subcriteria_3.extend(data_total[0].columns.to_list())
              dropdown_subcriteria_3 = dcc.Dropdown(
                  id='select-subcriteria-3',
                  options=select_subcriteria_3,
                  value=select_subcriteria_3[0],  # Select the first option as default
                  style={'position': 'relative',
                        'width': '100%',
                        'zIndex': 700,
                        'border': 'none',
                        'color': '#00782b',
                        'font-family': 'DM Sans !important' ,
                        'font-size': '20px'},
                )
              
              select_subcriteria_4 = ['']
              select_subcriteria_4.extend(data_total[0].columns.to_list())
              dropdown_subcriteria_4 = dcc.Dropdown(
                  id='select-subcriteria-4',
                  options=select_subcriteria_4,
                  value=select_subcriteria_4[0],  # Select the first option as default
                  style={'position': 'relative',
                        'width': '100%',
                        'zIndex': 600,
                        'border': 'none',
                        'color': '#00782b',
                        'font-family': 'DM Sans !important' ,
                        'font-size': '20px'},
                )
              
              select_target = ['']
              select_target.extend(data_total[0].columns.to_list())
              dropdown_target = dcc.Dropdown(
                  id='select-target',
                  options=select_target,
                  value=select_target[0],  # Select the first option as default
                  style={'position': 'relative',
                        'width': '100%',
                        'zIndex': 500,
                        'border': 'none',
                        'color': '#00782b',
                        'font-family': 'DM Sans !important' ,
                        'font-size': '20px'},
                )              
              
              return dropdown_subcriteria_1, dropdown_subcriteria_2, dropdown_subcriteria_3, dropdown_subcriteria_4, dropdown_target

        case 'Eventos Tramo':
              selection_criteria[0] = select_criteria
              select_subcriteria_1 = ['']
              select_subcriteria_1.extend(data_total[1].columns.to_list())
              dropdown_subcriteria_1 = dcc.Dropdown(
                  id='select-subcriteria-1',
                  options=select_subcriteria_1,
                  value=select_subcriteria_1[0],  # Select the first option as default
                  style={'position': 'relative',
                        'width': '100%',
                        'zIndex': 900,
                        'border': 'none',
                        'color': '#00782b',
                        'font-family': 'DM Sans !important' ,
                        'font-size': '20px'},
                )
              
              select_subcriteria_2 = ['']
              select_subcriteria_2.extend(data_total[1].columns.to_list())
              dropdown_subcriteria_2 = dcc.Dropdown(
                  id='select-subcriteria-2',
                  options=select_subcriteria_2,
                  value=select_subcriteria_2[0],  # Select the first option as default
                  style={'position': 'relative',
                        'width': '100%',
                        'zIndex': 800,
                        'border': 'none',
                        'color': '#00782b',
                        'font-family': 'DM Sans !important' ,
                        'font-size': '20px'},
                )
              
              select_subcriteria_3 = ['']
              select_subcriteria_3.extend(data_total[1].columns.to_list())
              dropdown_subcriteria_3 = dcc.Dropdown(
                  id='select-subcriteria-3',
                  options=select_subcriteria_3,
                  value=select_subcriteria_3[0],  # Select the first option as default
                  style={'position': 'relative',
                        'width': '100%',
                        'zIndex': 700,
                        'border': 'none',
                        'color': '#00782b',
                        'font-family': 'DM Sans !important' ,
                        'font-size': '20px'},
                )
              
              select_subcriteria_4 = ['']
              select_subcriteria_4.extend(data_total[1].columns.to_list())
              dropdown_subcriteria_4 = dcc.Dropdown(
                  id='select-subcriteria-4',
                  options=select_subcriteria_4,
                  value=select_subcriteria_4[0],  # Select the first option as default
                  style={'position': 'relative',
                        'width': '100%',
                        'zIndex': 600,
                        'border': 'none',
                        'color': '#00782b',
                        'font-family': 'DM Sans !important' ,
                        'font-size': '20px'},
                )
              
              select_target = ['']
              select_target.extend(data_total[1].columns.to_list())
              dropdown_target = dcc.Dropdown(
                  id='select-target',
                  options=select_target,
                  value=select_target[0],  # Select the first option as default
                  style={'position': 'relative',
                        'width': '100%',
                        'zIndex': 500,
                        'border': 'none',
                        'color': '#00782b',
                        'font-family': 'DM Sans !important' ,
                        'font-size': '20px'},
                )              
              
              return dropdown_subcriteria_1, dropdown_subcriteria_2, dropdown_subcriteria_3, dropdown_subcriteria_4, dropdown_target

        case 'Eventos Transformador':
              selection_criteria[0] = select_criteria
              select_subcriteria_1 = ['']
              select_subcriteria_1.extend(data_total[2].columns.to_list())
              dropdown_subcriteria_1 = dcc.Dropdown(
                  id='select-subcriteria-1',
                  options=select_subcriteria_1,
                  value=select_subcriteria_1[0],  # Select the first option as default
                  style={'position': 'relative',
                        'width': '100%',
                        'zIndex': 900,
                        'border': 'none',
                        'color': '#00782b',
                        'font-family': 'DM Sans !important' ,
                        'font-size': '20px'},
                )
              
              select_subcriteria_2 = ['']
              select_subcriteria_2.extend(data_total[2].columns.to_list())
              dropdown_subcriteria_2 = dcc.Dropdown(
                  id='select-subcriteria-2',
                  options=select_subcriteria_2,
                  value=select_subcriteria_2[0],  # Select the first option as default
                  style={'position': 'relative',
                        'width': '100%',
                        'zIndex': 800,
                        'border': 'none',
                        'color': '#00782b',
                        'font-family': 'DM Sans !important' ,
                        'font-size': '20px'},
                )
              
              select_subcriteria_3 = ['']
              select_subcriteria_3.extend(data_total[2].columns.to_list())
              dropdown_subcriteria_3 = dcc.Dropdown(
                  id='select-subcriteria-3',
                  options=select_subcriteria_3,
                  value=select_subcriteria_3[0],  # Select the first option as default
                  style={'position': 'relative',
                        'width': '100%',
                        'zIndex': 700,
                        'border': 'none',
                        'color': '#00782b',
                        'font-family': 'DM Sans !important' ,
                        'font-size': '20px'},
                )
              
              select_subcriteria_4 = ['']
              select_subcriteria_4.extend(data_total[2].columns.to_list())
              dropdown_subcriteria_4 = dcc.Dropdown(
                  id='select-subcriteria-4',
                  options=select_subcriteria_4,
                  value=select_subcriteria_4[0],  # Select the first option as default
                  style={'position': 'relative',
                        'width': '100%',
                        'zIndex': 600,
                        'border': 'none',
                        'color': '#00782b',
                        'font-family': 'DM Sans !important' ,
                        'font-size': '20px'},
                )
              
              select_target = ['']
              select_target.extend(data_total[2].columns.to_list())
              dropdown_target = dcc.Dropdown(
                  id='select-target',
                  options=select_target,
                  value=select_target[0],  # Select the first option as default
                  style={'position': 'relative',
                        'width': '100%',
                        'zIndex': 500,
                        'border': 'none',
                        'color': '#00782b',
                        'font-family': 'DM Sans !important' ,
                        'font-size': '20px'},
                )              
              
              return dropdown_subcriteria_1, dropdown_subcriteria_2, dropdown_subcriteria_3, dropdown_subcriteria_4, dropdown_target
            
        case _:
            selection_criteria[0] = ''
            return None, None, None, None, None
            
@app.callback(
    Output('sub-criteria-1-filters-container', 'children'),
    Input('select-subcriteria-1', 'value'),
)

def select_subcriteria_1(selection_sub_criteria_1):
        global selection_criteria

        match selection_criteria[0]:
        
                case 'Eventos Interruptor':
                        if selection_sub_criteria_1 != '':
                                match data_total[0][selection_sub_criteria_1].dtype:
                        
                                        case 'O' | 'int64':

                                                selection_criteria[1][0] = 'seleccion'
                                                selection_criteria[1][1] = selection_sub_criteria_1
                                                select_subcriteria = ['']
                                                select_subcriteria.extend(data_total[0][selection_sub_criteria_1].unique())

                                                div = [ 
                                                        html.Div('Selección:',className='sub-criteria-1-1-filter-text',
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
                                                        html.Div(className='sub-criteria-1-1-filter-container',
                                                        style={
                                                                'width': '50%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 9%'
                                                        }, children = [
                                                        dcc.Dropdown(
                                                                id='select-subcriteria-1-1',
                                                                options=select_subcriteria,
                                                                value=select_subcriteria[0],  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 850,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '20px'},
                                                                )
                                                        ])
                                                ]

                                                return div
                                
                                        case 'float32' | 'float64':
                                        
                                                selection_criteria[1][0] = 'rango_num'
                                                selection_criteria[1][1] = selection_sub_criteria_1

                                                div = [
                                                        html.Div('Operador:',className='sub-criteria-1-1-filter-text',
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
                                                        html.Div(className='sub-criteria-1-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 8%'
                                                                }, children=[
                                                                dcc.Dropdown(
                                                                                id='select-subcriteria-1-1',
                                                                                options=['', '>', '>=','<','<=','!=','=='],
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 850,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '20px'},
                                                                                )
                                                                ]),
                                                        html.Div('Valor:',className='sub-criteria-1-2-filter-text',
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
                                                        html.Div(className='sub-criteria-1-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 0%',
                                                                }, children = [
                                                                dcc.Input(
                                                                        id='select-subcriteria-1-2',
                                                                        type='number',
                                                                        placeholder='Ingresa un valor',
                                                                        style={'position': 'relative',
                                                                                'width': '91.0%',
                                                                                'zIndex': 850,
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
                                                selection_criteria[1][1] = selection_sub_criteria_1
                                                options_dates =  sorted(data_total[0][selection_sub_criteria_1].unique())
                                                options_dates = [date.strftime('%Y-%m-%d') for date in options_dates]
                                                options_dates = list(sorted(set(options_dates)))
                                        
                                                div = [ html.Div('Desde:',className='sub-criteria-1-1-filter-text',
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
                                                        html.Div(className='sub-criteria-1-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children = [
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-1-1',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 850,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),
                                                        html.Div('Hasta:',className='sub-criteria-1-2-filter-text',
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
                                                        html.Div(className='sub-criteria-1-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children=[
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-1-2',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 850,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),    
                                                ]
                                                
                                                return div

                                        case _:
                                                
                                                return None

                        elif selection_sub_criteria_1 == '':
                               selection_criteria[1][0] = ''
                               selection_criteria[1][1] = ''
                        
                        else:   
                                return dash.no_update

                case 'Eventos Tramo':
               
                        if selection_sub_criteria_1 != '':
                               
                                match data_total[1][selection_sub_criteria_1].dtype:
                        
                                        case 'O' | 'int64':

                                                selection_criteria[1][0] = 'seleccion'
                                                selection_criteria[1][1] = selection_sub_criteria_1
                                                select_subcriteria = ['']
                                                select_subcriteria.extend(data_total[1][selection_sub_criteria_1].unique())

                                                div = [ 
                                                        html.Div('Selección:',className='sub-criteria-1-1-filter-text',
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
                                                        html.Div(className='sub-criteria-1-1-filter-container',
                                                        style={
                                                        'width': '50%',
                                                        'height': '100%',
                                                        'borderRadius': '5px',
                                                        'backgroundColor': 'white',
                                                        'margin': '0 0 0 9%'
                                                        }, children = [
                                                        dcc.Dropdown(
                                                                id='select-subcriteria-1-1',
                                                                options=select_subcriteria,
                                                                value=select_subcriteria[0],  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 850,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '20px'},
                                                                )
                                                        ])
                                                ]

                                                return div
                                
                                        case 'float32' | 'float64':
                                        
                                                selection_criteria[1][0] = 'rango_num'
                                                selection_criteria[1][1] = selection_sub_criteria_1

                                                div = [
                                                        html.Div('Operador:',className='sub-criteria-1-1-filter-text',
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
                                                        html.Div(className='sub-criteria-1-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 8%'
                                                                }, children=[
                                                                dcc.Dropdown(
                                                                                id='select-subcriteria-1-1',
                                                                                options=['', '>', '>=','<','<=','!=','=='],
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 850,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '20px'},
                                                                                )
                                                                ]),
                                                        html.Div('Valor:',className='sub-criteria-1-2-filter-text',
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
                                                        html.Div(className='sub-criteria-1-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 0%',
                                                                }, children = [
                                                                dcc.Input(
                                                                        id='select-subcriteria-1-2',
                                                                        type='number',
                                                                        placeholder='Ingresa un valor',
                                                                        style={'position': 'relative',
                                                                                'width': '91.0%',
                                                                                'zIndex': 850,
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
                                                selection_criteria[1][1] = selection_sub_criteria_1
                                                options_dates =  sorted(data_total[1][selection_sub_criteria_1].unique())
                                                options_dates = [date.strftime('%Y-%m-%d') for date in options_dates]
                                                options_dates = list(sorted(set(options_dates)))
                                        
                                                div = [ html.Div('Desde:',className='sub-criteria-1-1-filter-text',
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
                                                        html.Div(className='sub-criteria-1-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children = [
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-1-1',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 850,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),
                                                        html.Div('Hasta:',className='sub-criteria-1-2-filter-text',
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
                                                        html.Div(className='sub-criteria-1-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children=[
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-1-2',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 850,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),    
                                                ]

                                                return div

                                        case _:
                                        
                                                return None

                        elif selection_sub_criteria_1 == '':
                               selection_criteria[1][0] = ''
                               selection_criteria[1][1] = ''
                        
                        else:
                                return dash.no_update

                case 'Eventos Transformador':
               
                        if selection_sub_criteria_1 != '':
                               
                                match data_total[2][selection_sub_criteria_1].dtype:
                        
                                        case 'O' | 'int64':

                                                selection_criteria[1][0] = 'seleccion'
                                                selection_criteria[1][1] = selection_sub_criteria_1
                                                select_subcriteria = ['']
                                                select_subcriteria.extend(data_total[2][selection_sub_criteria_1].unique())

                                                div = [ 
                                                        html.Div('Selección:',className='sub-criteria-1-1-filter-text',
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
                                                        html.Div(className='sub-criteria-1-1-filter-container',
                                                        style={
                                                        'width': '50%',
                                                        'height': '100%',
                                                        'borderRadius': '5px',
                                                        'backgroundColor': 'white',
                                                        'margin': '0 0 0 9%'
                                                        }, children = [
                                                        dcc.Dropdown(
                                                                id='select-subcriteria-1-1',
                                                                options=select_subcriteria,
                                                                value=select_subcriteria[0],  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 850,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '20px'},
                                                                )
                                                        ])
                                                ]

                                                return div
                                
                                        case 'float32' | 'float64':
                                        
                                                selection_criteria[1][0] = 'rango_num'
                                                selection_criteria[1][1] = selection_sub_criteria_1

                                                div = [
                                                        html.Div('Operador:',className='sub-criteria-1-1-filter-text',
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
                                                        html.Div(className='sub-criteria-1-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 8%'
                                                                }, children=[
                                                                dcc.Dropdown(
                                                                                id='select-subcriteria-1-1',
                                                                                options=['', '>', '>=','<','<=','!=','=='],
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 850,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '20px'},
                                                                                )
                                                                ]),
                                                        html.Div('Valor:',className='sub-criteria-1-2-filter-text',
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
                                                        html.Div(className='sub-criteria-1-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 0%',
                                                                }, children = [
                                                                dcc.Input(
                                                                        id='select-subcriteria-1-2',
                                                                        type='number',
                                                                        placeholder='Ingresa un valor',
                                                                        style={'position': 'relative',
                                                                                'width': '91.0%',
                                                                                'zIndex': 850,
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
                                                selection_criteria[1][1] = selection_sub_criteria_1
                                                options_dates =  sorted(data_total[2][selection_sub_criteria_1].unique())
                                                options_dates = [date.strftime('%Y-%m-%d') for date in options_dates]
                                                options_dates = list(sorted(set(options_dates)))
                                        
                                                div = [ html.Div('Desde:',className='sub-criteria-1-1-filter-text',
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
                                                        html.Div(className='sub-criteria-1-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children = [
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-1-1',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 850,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),
                                                        html.Div('Hasta:',className='sub-criteria-1-2-filter-text',
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
                                                        html.Div(className='sub-criteria-1-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children=[
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-1-2',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 850,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),    
                                                ]

                                                return div
        
                                        case _:
                                                
                                                return None

                        elif selection_sub_criteria_1 == '':
                               selection_criteria[1][0] = ''
                               selection_criteria[1][1] = ''
                        
                        else:
                                return dash.no_update

                case _:
                      selection_criteria[1][0] = ''
                      return dash.no_update

@app.callback(
    Input('select-subcriteria-1-1', 'value'),
)
def select_subcriteria_1_1(selection_sub_criteria_1_1):
        global selection_criteria
        selection_criteria[1][2] = selection_sub_criteria_1_1

@app.callback(
    Input('select-subcriteria-1-2', 'value'),
)
def select_subcriteria_1_2(selection_sub_criteria_1_2):
        global selection_criteria
        selection_criteria[1][3] = selection_sub_criteria_1_2




@app.callback(
    Output('sub-criteria-2-filters-container', 'children'),
    Input('select-subcriteria-2', 'value'),
)

def select_subcriteria_2(selection_sub_criteria_2):
        global selection_criteria
        match selection_criteria[0]:
        
                case 'Eventos Interruptor':
                        if selection_sub_criteria_2 != '':
                                match data_total[0][selection_sub_criteria_2].dtype:
                        
                                        case 'O' | 'int64':

                                                selection_criteria[2][0] = 'seleccion'
                                                selection_criteria[2][1] = selection_sub_criteria_2
                                                select_subcriteria = ['']
                                                select_subcriteria.extend(data_total[0][selection_sub_criteria_2].unique())

                                                div = [ 
                                                        html.Div('Selección:',className='sub-criteria-2-1-filter-text',
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
                                                        html.Div(className='sub-criteria-2-1-filter-container',
                                                        style={
                                                                'width': '50%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 9%'
                                                        }, children = [
                                                        dcc.Dropdown(
                                                                id='select-subcriteria-2-1',
                                                                options=select_subcriteria,
                                                                value=select_subcriteria[0],  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 750,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '20px'},
                                                                )
                                                        ])
                                                ]

                                                return div
                                
                                        case 'float32' | 'float64':
                                        
                                                selection_criteria[2][0] = 'rango_num'
                                                selection_criteria[2][1] = selection_sub_criteria_2

                                                div = [
                                                        html.Div('Operador:',className='sub-criteria-2-1-filter-text',
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
                                                        html.Div(className='sub-criteria-2-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 8%'
                                                                }, children=[
                                                                dcc.Dropdown(
                                                                                id='select-subcriteria-2-1',
                                                                                options=['', '>', '>=','<','<=','!=','=='],
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 750,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '20px'},
                                                                                )
                                                                ]),
                                                        html.Div('Valor:',className='sub-criteria-2-2-filter-text',
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
                                                        html.Div(className='sub-criteria-2-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 0%',
                                                                }, children = [
                                                                dcc.Input(
                                                                        id='select-subcriteria-2-2',
                                                                        type='number',
                                                                        placeholder='Ingresa un valor',
                                                                        style={'position': 'relative',
                                                                                'width': '91.0%',
                                                                                'zIndex': 750,
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
                                                selection_criteria[2][1] = selection_sub_criteria_2
                                                options_dates =  sorted(data_total[0][selection_sub_criteria_2].unique())
                                                options_dates = [date.strftime('%Y-%m-%d') for date in options_dates]
                                                options_dates = list(sorted(set(options_dates)))
                                        
                                                div = [ html.Div('Desde:',className='sub-criteria-2-1-filter-text',
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
                                                        html.Div(className='sub-criteria-2-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children = [
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-2-1',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 750,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),
                                                        html.Div('Hasta:',className='sub-criteria-2-2-filter-text',
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
                                                        html.Div(className='sub-criteria-2-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children=[
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-2-2',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 750,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),    
                                                ]
                                                
                                                return div

                                        case _:
                                        
                                                return None

                        elif selection_sub_criteria_2 == '':
                               selection_criteria[2][0] = ''
                               selection_criteria[2][1] = ''
                        
                        else:
                                return dash.no_update

                case 'Eventos Tramo':
               
                        if selection_sub_criteria_2 != '':
                               
                                match data_total[1][selection_sub_criteria_2].dtype:
                        
                                        case 'O' | 'int64':

                                                selection_criteria[2][0] = 'seleccion'
                                                selection_criteria[2][1] = selection_sub_criteria_2
                                                select_subcriteria = ['']
                                                select_subcriteria.extend(data_total[1][selection_sub_criteria_2].unique())

                                                div = [ 
                                                        html.Div('Selección:',className='sub-criteria-2-1-filter-text',
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
                                                        html.Div(className='sub-criteria-2-1-filter-container',
                                                        style={
                                                        'width': '50%',
                                                        'height': '100%',
                                                        'borderRadius': '5px',
                                                        'backgroundColor': 'white',
                                                        'margin': '0 0 0 9%'
                                                        }, children = [
                                                        dcc.Dropdown(
                                                                id='select-subcriteria-2-1',
                                                                options=select_subcriteria,
                                                                value=select_subcriteria[0],  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 750,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '20px'},
                                                                )
                                                        ])
                                                ]

                                                return div
                                
                                        case 'float32' | 'float64':
                                        
                                                selection_criteria[2][0] = 'rango_num'
                                                selection_criteria[2][1] = selection_sub_criteria_2

                                                div = [
                                                        html.Div('Operador:',className='sub-criteria-2-1-filter-text',
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
                                                        html.Div(className='sub-criteria-2-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 8%'
                                                                }, children=[
                                                                dcc.Dropdown(
                                                                                id='select-subcriteria-2-1',
                                                                                options=['', '>', '>=','<','<=','!=','=='],
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 750,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '20px'},
                                                                                )
                                                                ]),
                                                        html.Div('Valor:',className='sub-criteria-2-2-filter-text',
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
                                                        html.Div(className='sub-criteria-2-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 0%',
                                                                }, children = [
                                                                dcc.Input(
                                                                        id='select-subcriteria-2-2',
                                                                        type='number',
                                                                        placeholder='Ingresa un valor',
                                                                        style={'position': 'relative',
                                                                                'width': '91.0%',
                                                                                'zIndex': 750,
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
                                                selection_criteria[2][1] = selection_sub_criteria_2
                                                options_dates =  sorted(data_total[1][selection_sub_criteria_2].unique())
                                                options_dates = [date.strftime('%Y-%m-%d') for date in options_dates]
                                                options_dates = list(sorted(set(options_dates)))
                                        
                                                div = [ html.Div('Desde:',className='sub-criteria-2-1-filter-text',
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
                                                        html.Div(className='sub-criteria-2-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children = [
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-2-1',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 750,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),
                                                        html.Div('Hasta:',className='sub-criteria-2-2-filter-text',
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
                                                        html.Div(className='sub-criteria-2-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children=[
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-2-2',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 750,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),    
                                                ]

                                                return div

                                        case _:
                                        
                                                return None

                        elif selection_sub_criteria_2 == '':
                               selection_criteria[2][0] = ''
                               selection_criteria[2][1] = ''
                        
                        else:
                                selection_criteria[2][0] = ''
                                return dash.no_update

                case 'Eventos Transformador':
               
                        if selection_sub_criteria_2 != '':
                               
                                match data_total[2][selection_sub_criteria_2].dtype:
                        
                                        case 'O' | 'int64':

                                                selection_criteria[2][0] = 'seleccion'
                                                selection_criteria[2][1] = selection_sub_criteria_2
                                                select_subcriteria = ['']
                                                select_subcriteria.extend(data_total[2][selection_sub_criteria_2].unique())

                                                div = [ 
                                                        html.Div('Selección:',className='sub-criteria-2-1-filter-text',
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
                                                        html.Div(className='sub-criteria-2-1-filter-container',
                                                        style={
                                                        'width': '50%',
                                                        'height': '100%',
                                                        'borderRadius': '5px',
                                                        'backgroundColor': 'white',
                                                        'margin': '0 0 0 9%'
                                                        }, children = [
                                                        dcc.Dropdown(
                                                                id='select-subcriteria-2-1',
                                                                options=select_subcriteria,
                                                                value=select_subcriteria[0],  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 750,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '20px'},
                                                                )
                                                        ])
                                                ]

                                                return div
                                
                                        case 'float32' | 'float64':
                                        
                                                selection_criteria[2][0] = 'rango_num'
                                                selection_criteria[2][1] = selection_sub_criteria_2

                                                div = [
                                                        html.Div('Operador:',className='sub-criteria-2-1-filter-text',
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
                                                        html.Div(className='sub-criteria-2-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 8%'
                                                                }, children=[
                                                                dcc.Dropdown(
                                                                                id='select-subcriteria-2-1',
                                                                                options=['', '>', '>=','<','<=','!=','=='],
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 750,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '20px'},
                                                                                )
                                                                ]),
                                                        html.Div('Valor:',className='sub-criteria-2-2-filter-text',
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
                                                        html.Div(className='sub-criteria-2-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 0%',
                                                                }, children = [
                                                                dcc.Input(
                                                                        id='select-subcriteria-2-2',
                                                                        type='number',
                                                                        placeholder='Ingresa un valor',
                                                                        style={'position': 'relative',
                                                                                'width': '91.0%',
                                                                                'zIndex': 750,
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
                                                selection_criteria[2][1] = selection_sub_criteria_2
                                                options_dates =  sorted(data_total[2][selection_sub_criteria_2].unique())
                                                options_dates = [date.strftime('%Y-%m-%d') for date in options_dates]
                                                options_dates = list(sorted(set(options_dates)))
                                        
                                                div = [ html.Div('Desde:',className='sub-criteria-2-1-filter-text',
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
                                                        html.Div(className='sub-criteria-2-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children = [
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-2-1',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 750,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),
                                                        html.Div('Hasta:',className='sub-criteria-2-2-filter-text',
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
                                                        html.Div(className='sub-criteria-2-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children=[
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-2-2',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 750,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),    
                                                ]
        
                                                return div

                                        case _:
                                                
                                                return None

                        elif selection_sub_criteria_2 == '':
                               selection_criteria[2][0] = ''
                               selection_criteria[2][1] = ''
                        
                        else:
                                return dash.no_update

                case _:
                      selection_criteria[2][0] = ''
                      return dash.no_update

@app.callback(
    Input('select-subcriteria-2-1', 'value'),
)
def select_subcriteria_2_1(selection_sub_criteria_2_1):
        global selection_criteria
        selection_criteria[2][2] = selection_sub_criteria_2_1

@app.callback(
    Input('select-subcriteria-2-2', 'value'),
)
def select_subcriteria_2_2(selection_sub_criteria_2_2):
        global selection_criteria
        selection_criteria[2][3] = selection_sub_criteria_2_2




@app.callback(
    Output('sub-criteria-3-filters-container', 'children'),
    Input('select-subcriteria-3', 'value'),
)

def select_subcriteria_3(selection_sub_criteria_3):
        global selection_criteria
        match selection_criteria[0]:
        
                case 'Eventos Interruptor':
                        if selection_sub_criteria_3 != '':
                                match data_total[0][selection_sub_criteria_3].dtype:
                        
                                        case 'O' | 'int64':

                                                selection_criteria[3][0] = 'seleccion'
                                                selection_criteria[3][1] = selection_sub_criteria_3
                                                select_subcriteria = ['']
                                                select_subcriteria.extend(data_total[0][selection_sub_criteria_3].unique())

                                                div = [ 
                                                        html.Div('Selección:',className='sub-criteria-3-1-filter-text',
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
                                                        html.Div(className='sub-criteria-3-1-filter-container',
                                                        style={
                                                                'width': '50%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 9%'
                                                        }, children = [
                                                        dcc.Dropdown(
                                                                id='select-subcriteria-3-1',
                                                                options=select_subcriteria,
                                                                value=select_subcriteria[0],  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 650,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '20px'},
                                                                )
                                                        ])
                                                ]

                                                return div
                                
                                        case 'float32' | 'float64':
                                        
                                                selection_criteria[3][0] = 'rango_num'
                                                selection_criteria[3][1] = selection_sub_criteria_3

                                                div = [
                                                        html.Div('Operador:',className='sub-criteria-3-1-filter-text',
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
                                                        html.Div(className='sub-criteria-3-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 8%'
                                                                }, children=[
                                                                dcc.Dropdown(
                                                                                id='select-subcriteria-3-1',
                                                                                options=['', '>', '>=','<','<=','!=','=='],
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 650,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '20px'},
                                                                                )
                                                                ]),
                                                        html.Div('Valor:',className='sub-criteria-3-2-filter-text',
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
                                                        html.Div(className='sub-criteria-3-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 0%',
                                                                }, children = [
                                                                dcc.Input(
                                                                        id='select-subcriteria-3-2',
                                                                        type='number',
                                                                        placeholder='Ingresa un valor',
                                                                        style={'position': 'relative',
                                                                                'width': '91.0%',
                                                                                'zIndex': 650,
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
                                                selection_criteria[3][1] = selection_sub_criteria_3
                                                options_dates =  sorted(data_total[0][selection_sub_criteria_3].unique())
                                                options_dates = [date.strftime('%Y-%m-%d') for date in options_dates]
                                                options_dates = list(sorted(set(options_dates)))
                                        
                                                div = [ html.Div('Desde:',className='sub-criteria-3-1-filter-text',
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
                                                        html.Div(className='sub-criteria-3-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children = [
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-3-1',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 650,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),
                                                        html.Div('Hasta:',className='sub-criteria-3-2-filter-text',
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
                                                        html.Div(className='sub-criteria-3-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children=[
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-3-2',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 650,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),    
                                                ]
                                                
                                                return div

                                        case _:
                                        
                                                return None

                        elif selection_sub_criteria_3 == '':
                               selection_criteria[3][0] = ''
                               selection_criteria[3][1] = ''
                        
                        else:
                                return dash.no_update

                case 'Eventos Tramo':
               
                        if selection_sub_criteria_3 != '':
                               
                                match data_total[1][selection_sub_criteria_3].dtype:
                        
                                        case 'O' | 'int64':

                                                selection_criteria[3][0] = 'seleccion'
                                                selection_criteria[3][1] = selection_sub_criteria_3
                                                select_subcriteria = ['']
                                                select_subcriteria.extend(data_total[1][selection_sub_criteria_3].unique())

                                                div = [ 
                                                        html.Div('Selección:',className='sub-criteria-3-1-filter-text',
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
                                                        html.Div(className='sub-criteria-3-1-filter-container',
                                                        style={
                                                        'width': '50%',
                                                        'height': '100%',
                                                        'borderRadius': '5px',
                                                        'backgroundColor': 'white',
                                                        'margin': '0 0 0 9%'
                                                        }, children = [
                                                        dcc.Dropdown(
                                                                id='select-subcriteria-3-1',
                                                                options=select_subcriteria,
                                                                value=select_subcriteria[0],  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 650,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '20px'},
                                                                )
                                                        ])
                                                ]

                                                return div
                                
                                        case 'float32' | 'float64':
                                        
                                                selection_criteria[3][0] = 'rango_num'
                                                selection_criteria[3][1] = selection_sub_criteria_3

                                                div = [
                                                        html.Div('Operador:',className='sub-criteria-3-1-filter-text',
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
                                                        html.Div(className='sub-criteria-3-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 8%'
                                                                }, children=[
                                                                dcc.Dropdown(
                                                                                id='select-subcriteria-3-1',
                                                                                options=['', '>', '>=','<','<=','!=','=='],
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 650,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '20px'},
                                                                                )
                                                                ]),
                                                        html.Div('Valor:',className='sub-criteria-3-2-filter-text',
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
                                                        html.Div(className='sub-criteria-3-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 0%',
                                                                }, children = [
                                                                dcc.Input(
                                                                        id='select-subcriteria-3-2',
                                                                        type='number',
                                                                        placeholder='Ingresa un valor',
                                                                        style={'position': 'relative',
                                                                                'width': '91.0%',
                                                                                'zIndex': 650,
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
                                                selection_criteria[3][1] = selection_sub_criteria_3
                                                options_dates =  sorted(data_total[1][selection_sub_criteria_3].unique())
                                                options_dates = [date.strftime('%Y-%m-%d') for date in options_dates]
                                                options_dates = list(sorted(set(options_dates)))
                                        
                                                div = [ html.Div('Desde:',className='sub-criteria-3-1-filter-text',
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
                                                        html.Div(className='sub-criteria-3-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children = [
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-3-1',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 650,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),
                                                        html.Div('Hasta:',className='sub-criteria-3-2-filter-text',
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
                                                        html.Div(className='sub-criteria-3-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children=[
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-3-2',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 650,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),    
                                                ]

                                                return div

                                        case _:
                                        
                                                return None

                        elif selection_sub_criteria_3 == '':
                               selection_criteria[3][0] = ''
                               selection_criteria[3][1] = ''
                        
                        else:
                                return dash.no_update

                case 'Eventos Transformador':
               
                        if selection_sub_criteria_3 != '':
                               
                                match data_total[2][selection_sub_criteria_3].dtype:
                        
                                        case 'O' | 'int64':

                                                selection_criteria[3][0] = 'seleccion'
                                                selection_criteria[3][1] = selection_sub_criteria_3
                                                select_subcriteria = ['']
                                                select_subcriteria.extend(data_total[2][selection_sub_criteria_3].unique())

                                                div = [ 
                                                        html.Div('Selección:',className='sub-criteria-3-1-filter-text',
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
                                                        html.Div(className='sub-criteria-3-1-filter-container',
                                                        style={
                                                        'width': '50%',
                                                        'height': '100%',
                                                        'borderRadius': '5px',
                                                        'backgroundColor': 'white',
                                                        'margin': '0 0 0 9%'
                                                        }, children = [
                                                        dcc.Dropdown(
                                                                id='select-subcriteria-3-1',
                                                                options=select_subcriteria,
                                                                value=select_subcriteria[0],  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 850,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '20px'},
                                                                )
                                                        ])
                                                ]

                                                return div
                                
                                        case 'float32' | 'float64':
                                        
                                                selection_criteria[3][0] = 'rango_num'
                                                selection_criteria[3][1] = selection_sub_criteria_3

                                                div = [
                                                        html.Div('Operador:',className='sub-criteria-3-1-filter-text',
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
                                                        html.Div(className='sub-criteria-3-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 8%'
                                                                }, children=[
                                                                dcc.Dropdown(
                                                                                id='select-subcriteria-3-1',
                                                                                options=['', '>', '>=','<','<=','!=','=='],
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 650,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '20px'},
                                                                                )
                                                                ]),
                                                        html.Div('Valor:',className='sub-criteria-3-2-filter-text',
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
                                                        html.Div(className='sub-criteria-3-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 0%',
                                                                }, children = [
                                                                dcc.Input(
                                                                        id='select-subcriteria-3-2',
                                                                        type='number',
                                                                        placeholder='Ingresa un valor',
                                                                        style={'position': 'relative',
                                                                                'width': '91.0%',
                                                                                'zIndex': 650,
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
                                                selection_criteria[3][1] = selection_sub_criteria_3
                                                options_dates =  sorted(data_total[2][selection_sub_criteria_3].unique())
                                                options_dates = [date.strftime('%Y-%m-%d') for date in options_dates]
                                                options_dates = list(sorted(set(options_dates)))
                                        
                                                div = [ html.Div('Desde:',className='sub-criteria-3-1-filter-text',
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
                                                        html.Div(className='sub-criteria-3-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children = [
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-3-1',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 650,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),
                                                        html.Div('Hasta:',className='sub-criteria-3-2-filter-text',
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
                                                        html.Div(className='sub-criteria-3-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children=[
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-3-2',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 650,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),    
                                                ]
        
                                                return div

                                        case _:
                                                
                                                return None

                        elif selection_sub_criteria_3 == '':
                               selection_criteria[3][0] = ''
                               selection_criteria[3][1] = ''
                        
                        else:
                                return dash.no_update

                case _:
                      selection_criteria[3][0] = ''
                      return dash.no_update

@app.callback(
    Input('select-subcriteria-3-1', 'value'),
)
def select_subcriteria_3_1(selection_sub_criteria_3_1):
        global selection_criteria
        selection_criteria[3][2] = selection_sub_criteria_3_1

@app.callback(
    Input('select-subcriteria-3-2', 'value'),
)
def select_subcriteria_3_2(selection_sub_criteria_3_2):
        global selection_criteria
        selection_criteria[3][3] = selection_sub_criteria_3_2




@app.callback(
    Output('sub-criteria-4-filters-container', 'children'),
    Input('select-subcriteria-4', 'value'),
)

def select_subcriteria_4(selection_sub_criteria_4):
        global selection_criteria
        match selection_criteria[0]:
        
                case 'Eventos Interruptor':
                        if selection_sub_criteria_4 != '':
                                match data_total[0][selection_sub_criteria_4].dtype:
                        
                                        case 'O' | 'int64':

                                                selection_criteria[4][0] = 'seleccion'
                                                selection_criteria[4][1] = selection_sub_criteria_4
                                                select_subcriteria = ['']
                                                select_subcriteria.extend(data_total[0][selection_sub_criteria_4].unique())

                                                div = [ 
                                                        html.Div('Selección:',className='sub-criteria-4-1-filter-text',
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
                                                        html.Div(className='sub-criteria-4-1-filter-container',
                                                        style={
                                                                'width': '50%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 9%'
                                                        }, children = [
                                                        dcc.Dropdown(
                                                                id='select-subcriteria-4-1',
                                                                options=select_subcriteria,
                                                                value=select_subcriteria[0],  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 550,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '20px'},
                                                                )
                                                        ])
                                                ]

                                                return div
                                
                                        case 'float32' | 'float64':
                                        
                                                selection_criteria[4][0] = 'rango_num'
                                                selection_criteria[4][1] = selection_sub_criteria_4

                                                div = [
                                                        html.Div('Operador:',className='sub-criteria-4-1-filter-text',
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
                                                        html.Div(className='sub-criteria-4-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 8%'
                                                                }, children=[
                                                                dcc.Dropdown(
                                                                                id='select-subcriteria-4-1',
                                                                                options=['', '>', '>=','<','<=','!=','=='],
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 550,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '20px'},
                                                                                )
                                                                ]),
                                                        html.Div('Valor:',className='sub-criteria-4-2-filter-text',
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
                                                        html.Div(className='sub-criteria-4-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 0%',
                                                                }, children = [
                                                                dcc.Input(
                                                                        id='select-subcriteria-4-2',
                                                                        type='number',
                                                                        placeholder='Ingresa un valor',
                                                                        style={'position': 'relative',
                                                                                'width': '91.0%',
                                                                                'zIndex': 550,
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

                                                selection_criteria[4][0] = 'fecha'
                                                selection_criteria[4][1] = selection_sub_criteria_4
                                                options_dates =  sorted(data_total[0][selection_sub_criteria_4].unique())
                                                options_dates = [date.strftime('%Y-%m-%d') for date in options_dates]
                                                options_dates = list(sorted(set(options_dates)))
                                        
                                                div = [ html.Div('Desde:',className='sub-criteria-4-1-filter-text',
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
                                                        html.Div(className='sub-criteria-4-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children = [
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-4-1',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 550,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),
                                                        html.Div('Hasta:',className='sub-criteria-4-2-filter-text',
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
                                                        html.Div(className='sub-criteria-4-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children=[
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-4-2',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 550,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),    
                                                ]
                                                
                                                return div

                                        case _:
                                        
                                                return None

                        elif selection_sub_criteria_4 == '':
                               selection_criteria[4][0] = ''
                               selection_criteria[4][1] = ''
                        
                        else:
                                return dash.no_update

                case 'Eventos Tramo':
               
                        if selection_sub_criteria_4 != '':
                               
                                match data_total[1][selection_sub_criteria_4].dtype:
                        
                                        case 'O' | 'int64':

                                                selection_criteria[4][0] = 'seleccion'
                                                selection_criteria[4][1] = selection_sub_criteria_4
                                                select_subcriteria = ['']
                                                select_subcriteria.extend(data_total[1][selection_sub_criteria_4].unique())

                                                div = [ 
                                                        html.Div('Selección:',className='sub-criteria-4-1-filter-text',
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
                                                        html.Div(className='sub-criteria-4-1-filter-container',
                                                        style={
                                                        'width': '50%',
                                                        'height': '100%',
                                                        'borderRadius': '5px',
                                                        'backgroundColor': 'white',
                                                        'margin': '0 0 0 9%'
                                                        }, children = [
                                                        dcc.Dropdown(
                                                                id='select-subcriteria-4-1',
                                                                options=select_subcriteria,
                                                                value=select_subcriteria[0],  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 850,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '20px'},
                                                                )
                                                        ])
                                                ]

                                                return div
                                
                                        case 'float32' | 'float64':
                                        
                                                selection_criteria[4][0] = 'rango_num'
                                                selection_criteria[4][1] = selection_sub_criteria_4

                                                div = [
                                                        html.Div('Operador:',className='sub-criteria-4-1-filter-text',
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
                                                        html.Div(className='sub-criteria-4-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 8%'
                                                                }, children=[
                                                                dcc.Dropdown(
                                                                                id='select-subcriteria-4-1',
                                                                                options=['', '>', '>=','<','<=','!=','=='],
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 550,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '20px'},
                                                                                )
                                                                ]),
                                                        html.Div('Valor:',className='sub-criteria-4-2-filter-text',
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
                                                        html.Div(className='sub-criteria-4-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 0%',
                                                                }, children = [
                                                                dcc.Input(
                                                                        id='select-subcriteria-4-2',
                                                                        type='number',
                                                                        placeholder='Ingresa un valor',
                                                                        style={'position': 'relative',
                                                                                'width': '91.0%',
                                                                                'zIndex': 550,
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

                                                selection_criteria[4][0] = 'fecha'
                                                selection_criteria[4][1] = selection_sub_criteria_4
                                                options_dates =  sorted(data_total[1][selection_sub_criteria_4].unique())
                                                options_dates = [date.strftime('%Y-%m-%d') for date in options_dates]
                                                options_dates = list(sorted(set(options_dates)))
                                        
                                                div = [ html.Div('Desde:',className='sub-criteria-4-1-filter-text',
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
                                                        html.Div(className='sub-criteria-4-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children = [
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-4-1',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 550,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),
                                                        html.Div('Hasta:',className='sub-criteria-4-2-filter-text',
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
                                                        html.Div(className='sub-criteria-4-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children=[
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-4-2',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 550,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),    
                                                ]

                                                return div

                                        case _:
                                        
                                                return None

                        elif selection_sub_criteria_4 == '':
                               selection_criteria[4][0] = ''
                               selection_criteria[4][1] = ''
                        
                        else:
                                return dash.no_update

                case 'Eventos Transformador':
               
                        if selection_sub_criteria_4 != '':
                               
                                match data_total[2][selection_sub_criteria_4].dtype:
                        
                                        case 'O' | 'int64':

                                                selection_criteria[4][0] = 'seleccion'
                                                selection_criteria[4][1] = selection_sub_criteria_4
                                                select_subcriteria = ['']
                                                select_subcriteria.extend(data_total[2][selection_sub_criteria_4].unique())

                                                div = [ 
                                                        html.Div('Selección:',className='sub-criteria-4-1-filter-text',
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
                                                        html.Div(className='sub-criteria-4-1-filter-container',
                                                        style={
                                                        'width': '50%',
                                                        'height': '100%',
                                                        'borderRadius': '5px',
                                                        'backgroundColor': 'white',
                                                        'margin': '0 0 0 9%'
                                                        }, children = [
                                                        dcc.Dropdown(
                                                                id='select-subcriteria-4-1',
                                                                options=select_subcriteria,
                                                                value=select_subcriteria[0],  # Select the first option as default
                                                                style={'position': 'relative',
                                                                        'width': '100%',
                                                                        'zIndex': 550,
                                                                        'border': 'none',
                                                                        'color': '#00782b',
                                                                        'font-family': 'DM Sans !important' ,
                                                                        'font-size': '20px'},
                                                                )
                                                        ])
                                                ]

                                                return div
                                
                                        case 'float32' | 'float64':
                                        
                                                selection_criteria[4][0] = 'rango_num'
                                                selection_criteria[4][1] = selection_sub_criteria_4

                                                div = [
                                                        html.Div('Operador:',className='sub-criteria-4-1-filter-text',
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
                                                        html.Div(className='sub-criteria-4-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 8%'
                                                                }, children=[
                                                                dcc.Dropdown(
                                                                                id='select-subcriteria-4-1',
                                                                                options=['', '>', '>=','<','<=','!=','=='],
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 550,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '20px'},
                                                                                )
                                                                ]),
                                                        html.Div('Valor:',className='sub-criteria-4-2-filter-text',
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
                                                        html.Div(className='sub-criteria-4-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 0%',
                                                                }, children = [
                                                                dcc.Input(
                                                                        id='select-subcriteria-4-2',
                                                                        type='number',
                                                                        placeholder='Ingresa un valor',
                                                                        style={'position': 'relative',
                                                                                'width': '91.0%',
                                                                                'zIndex': 550,
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

                                                selection_criteria[4][0] = 'fecha'
                                                selection_criteria[4][1] = selection_sub_criteria_4
                                                options_dates =  sorted(data_total[2][selection_sub_criteria_4].unique())
                                                options_dates = [date.strftime('%Y-%m-%d') for date in options_dates]
                                                options_dates = list(sorted(set(options_dates)))
                                        
                                                div = [ html.Div('Desde:',className='sub-criteria-4-1-filter-text',
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
                                                        html.Div(className='sub-criteria-4-1-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children = [
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-4-1',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 550,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),
                                                        html.Div('Hasta:',className='sub-criteria-4-2-filter-text',
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
                                                        html.Div(className='sub-criteria-4-2-filter-container',
                                                                style={
                                                                'width': '20%',
                                                                'height': '100%',
                                                                'borderRadius': '5px',
                                                                'backgroundColor': 'white',
                                                                'margin': '0 0 0 3%'
                                                                }, children=[
                                                                        dcc.Dropdown(
                                                                                id='select-subcriteria-4-2',
                                                                                options=options_dates,
                                                                                value='',  # Select the first option as default
                                                                                style={'position': 'relative',
                                                                                        'width': '100%',
                                                                                        'zIndex': 550,
                                                                                        'border': 'none',
                                                                                        'color': '#00782b',
                                                                                        'font-family': 'DM Sans !important' ,
                                                                                        'font-size': '15px'},
                                                                        )
                                                                ]),    
                                                ]
        
                                                return div

                                        case _:
                                                
                                                return None

                        elif selection_sub_criteria_4 == '':
                               selection_criteria[4][0] = ''
                               selection_criteria[4][1] = ''
                        
                        else:
                                selection_criteria[4][0] = ''
                                return dash.no_update

                case _:
                      
                      return None

@app.callback(
    Input('select-subcriteria-4-1', 'value'),
)
def select_subcriteria_4_1(selection_sub_criteria_4_1):
        global selection_criteria
        selection_criteria[4][2] = selection_sub_criteria_4_1

@app.callback(
    Input('select-subcriteria-4-2', 'value'),
)
def select_subcriteria_4_2(selection_sub_criteria_4_2):
        global selection_criteria
        selection_criteria[4][3] = selection_sub_criteria_4_2


@app.callback(
    Input('select-target', 'value'),
)
def select_target_fn(selection_target):
        global selection_criteria
        selection_criteria[5] = selection_target

                
@app.callback(
    [Output('probability_text', 'children'),
     Output('graph-fig-container', 'children')],
    Input('confirm-button-ok', 'n_clicks')   
)
def confirm_button_fn(n_clicks):
    global count_clicks
    global count
    
    # Si no se ha hecho clic aún (n_clicks es None)
    if n_clicks is None:
        return dash.no_update, dash.no_update
    
    # Si no ha habido un cambio de clics
    if n_clicks <= count_clicks:
        return dash.no_update, dash.no_update

    # Actualiza el contador de clics
    count_clicks = n_clicks
    
    # Verificar condiciones para la generación del texto
    if (selection_criteria[0] != '') and (selection_criteria[-1][0] != ''):
        probability_text = 'P(' + selection_criteria[-1] + ' | ' + selection_criteria[0]
        
        for i in range(1, 5):
            match selection_criteria[i][0]:
                case 'seleccion':
                    list_operators = ['', '>', '>=', '<', '<=', '!=', '==']
                    if (selection_criteria[i][2] not in list_operators and 
                        str(selection_criteria[i][2])[0] not in list_operators):
                        probability_text = probability_text + ', ' + selection_criteria[i][1] + ' = ' + str(selection_criteria[i][2])
                    else:
                        probability_text = probability_text + ', ' + selection_criteria[i][1] + ' ' + str(selection_criteria[i][2]) + ' ' + selection_criteria[i][3]
                
                case 'rango_num':
                    probability_text = probability_text + ', ' + selection_criteria[i][1] + ' ' + selection_criteria[i][2] + ' ' + str(selection_criteria[i][3])
                
                case 'fecha':
                    probability_text = probability_text + ', ' + selection_criteria[i][1] + ' ' + selection_criteria[i][2] + ' - ' + selection_criteria[i][3]
                
                case _:
                    pass
        
        probability_text = probability_text + ')'

        # Llamada a la función de generación de gráfico
        graph_probabilty(selection_criteria, data_total, probability_text, count)

        # Estilo del gráfico
        graph = (html.Img(src="/outputs/probability_graph_"+str(count)+".png",style={
                                'position': 'absolute',
                                'width': '100%',
                                'borderRadius': '10px',
                            }),
                dbc.Button(
                        html.I(className="save-button"),  # Flecha hacia la izquierda
                        id="save-button", color="primary", outline=True, style={
                        'width': '5vh', 
                        'position': 'absolute', 
                        'height': '5vh', 
                        'margin': '0 1% 0 0', 
                        'right': '1%',
                        'top': '2%',
                        'backgroundImage': "url('/assets/images/download-svgrepo-com.svg')",
                        'backgroundSize': 'cover',
                        'backgroundPosition': 'center',
                        'backgroundRepeat': 'no-repeat',
                        'border': 'none',
                        'backgroundColor': 'transparent',
                        'cursor': 'pointer'}
                        ),
                dcc.Download(id="download-file")
                )
        
        
        if count > 0:
               
               os.remove("./outputs/probability_graph_"+str(count-1)+".png")
        
        count = count + 1

        return probability_text, graph

    # Si no se cumplen las condiciones
    return dash.no_update, dash.no_update

@app.callback(
    Output("download-file", "data"),
    Input("save-button", "n_clicks"),
    prevent_initial_call=True
)
def download_file(n_clicks):
    # Generar el archivo para descarga
    return dcc.send_file("./outputs/probability_graph_"+str(count-1)+".png")

@app.callback(
    Output('url-graphs', 'pathname'),
    [Input('button-chat', 'n_clicks'),
     Input('button-maps', 'n_clicks'),
     Input('button-tab-net', 'n_clicks')]
)
def redirect_to_pages(n_clicks_chat, n_clicks_maps, n_clicks_tab_net):
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
    elif triggered_id == 'button-tab-net' and n_clicks_tab_net:
        return "/tab-net_page"

    raise exceptions.PreventUpdate