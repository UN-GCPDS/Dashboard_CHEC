import json
from dash import html, dcc, Output, Input
from dash import Dash, html
import dash_bootstrap_components as dbc

# Función que genera el contenido del espacio de trabajo
def work_space():

    with open("./options/criterias_1.json", "r", encoding='utf-8') as file:
        options_criterias = json.load(file)    

    # Puede ser un solo elemento o una lista de elementos
    return [
            # Selector de criterios
            html.Div(className='criteria-container', style={
                'position': 'relative',
                'width': '30%',
                'height': '95%',
                'backgroundColor': '#16D622',
                'margin': '0 0 0 1.5%',
                'borderRadius': '10px',
                'opacity': '0.7',
                'display': 'flex',
                'flexDirection': 'column',
                'alignItems': 'center',
                }, children=[
                    html.Div('Criterio',className='text-criteria', style={
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
                        'margin': '1vh 0 0 0'
                    }),
                    html.Div(className='criteria-container', style={
                        'width': '70%',
                        'height': '4%',
                        'borderRadius': '5px', 
                    }, children=[
                        dcc.Dropdown(
                            id='select-criteria',
                            options=options_criterias,
                            value=options_criterias[0]['value'],  # Select the first option as default
                        style={'position': 'relative',
                            'width': '100%',
                            'height': '4%',
                            'zIndex': 1000,
                            'border': 'none',
                            'color': '#00782b',
                            'font-family': 'DM Sans !important' ,
                            'font-size': '20px'},
                        )
                    ]),
                    html.Div(className='sub-criteria-container-1', style={
                        'margin': '1% 0 0 0',
                        'width': '100%',
                        'height': '18%',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'alignItems': 'center',
                        
                    }, children=[
                        html.Div('Sub-criterio 1',className='sub-criteria-1-text',
                                style={
                                    'width': '100%',
                                    'height': '28%',
                                    'lineHeight': '13vh',
                                    'color': '#FFFFFF',
                                    'fontFamily': "'DM Sans', sans-serif",
                                    'fontSize': '25px',
                                    'fontWeight': '700',
                                    'textAlign': 'center',
                                    'display': 'flex',
                                    'alignItems': 'center',
                                    'justifyContent': 'center',
                                    'margin': '3% 0 0 0'
                                }),
                        html.Div(id='sub-criteria-1-container',className='sub-criteria-1-container',
                                style={
                                    'width': '70%',
                                    'height': '28%',
                                    'borderRadius': '5px',
                                    'backgroundColor': 'white',
                                }),
                        html.Div(id='sub-criteria-1-filters-container',className='sub-criteria-1-filters-container',
                                style={
                                    'width': '100%',
                                    'height': '28%',
                                    'display': 'flex',
                                    'flexDirection': 'row',
                                    'alignItems': 'center',
                                    'justifyContent': 'center',
                                    'margin': '3% 0 0 0',
                                }),        
                    ]),                    
                    html.Div(className='sub-criteria-container-2', style={
                        'margin': '1% 0 0 0',
                        'width': '100%',
                        'height': '18%',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'alignItems': 'center',
                        
                    }, children=[
                        html.Div('Sub-criterio 2',className='sub-criteria-2-text',
                                style={
                                    'width': '100%',
                                    'height': '28%',
                                    'lineHeight': '13vh',
                                    'color': '#FFFFFF',
                                    'fontFamily': "'DM Sans', sans-serif",
                                    'fontSize': '25px',
                                    'fontWeight': '700',
                                    'textAlign': 'center',
                                    'display': 'flex',
                                    'alignItems': 'center',
                                    'justifyContent': 'center',
                                    'margin': '3% 0 0 0'
                                }),
                        html.Div(id='sub-criteria-2-container',className='sub-criteria-2-container',
                                style={
                                    'width': '70%',
                                    'height': '28%',
                                    'borderRadius': '5px',
                                    'backgroundColor': 'white',
                                }),
                        html.Div(id='sub-criteria-2-filters-container',className='sub-criteria-2-filters-container',
                                style={
                                    'width': '100%',
                                    'height': '28%',
                                    'display': 'flex',
                                    'flexDirection': 'row',
                                    'alignItems': 'center',
                                    'justifyContent': 'center',
                                    'margin': '3% 0 0 0',
                                }),        
                    ]),                    
                    html.Div(className='sub-criteria-container-3', style={
                        'margin': '1% 0 0 0',
                        'width': '100%',
                        'height': '18%',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'alignItems': 'center',
                        
                    }, children=[
                        html.Div('Sub-criterio 3',className='sub-criteria-3-text',
                                style={
                                    'width': '100%',
                                    'height': '28%',
                                    'lineHeight': '13vh',
                                    'color': '#FFFFFF',
                                    'fontFamily': "'DM Sans', sans-serif",
                                    'fontSize': '25px',
                                    'fontWeight': '700',
                                    'textAlign': 'center',
                                    'display': 'flex',
                                    'alignItems': 'center',
                                    'justifyContent': 'center',
                                    'margin': '3% 0 0 0'
                                }),
                        html.Div(id='sub-criteria-3-container',className='sub-criteria-3-container',
                                style={
                                    'width': '70%',
                                    'height': '28%',
                                    'borderRadius': '5px',
                                    'backgroundColor': 'white',
                                }),
                        html.Div(id='sub-criteria-3-filters-container',className='sub-criteria-3-filters-container',
                                style={
                                    'width': '100%',
                                    'height': '28%',
                                    'display': 'flex',
                                    'flexDirection': 'row',
                                    'alignItems': 'center',
                                    'justifyContent': 'center',
                                    'margin': '3% 0 0 0',
                                }),        
                    ]),                    
                    html.Div(className='sub-criteria-container-4', style={
                        'margin': '1% 0 0 0',
                        'width': '100%',
                        'height': '18%',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'alignItems': 'center',
                        
                    }, children=[
                        html.Div('Sub-criterio 4',className='sub-criteria-4-text',
                                style={
                                    'width': '100%',
                                    'height': '28%',
                                    'lineHeight': '13vh',
                                    'color': '#FFFFFF',
                                    'fontFamily': "'DM Sans', sans-serif",
                                    'fontSize': '25px',
                                    'fontWeight': '700',
                                    'textAlign': 'center',
                                    'display': 'flex',
                                    'alignItems': 'center',
                                    'justifyContent': 'center',
                                    'margin': '3% 0 0 0'
                                }),
                        html.Div(id='sub-criteria-4-container',className='sub-criteria-4-container',
                                style={
                                    'width': '70%',
                                    'height': '28%',
                                    'borderRadius': '5px',
                                    'backgroundColor': 'white',
                                }),
                        html.Div(id='sub-criteria-4-filters-container',className='sub-criteria-4-filters-container',
                                style={
                                    'width': '100%',
                                    'height': '28%',
                                    'display': 'flex',
                                    'flexDirection': 'row',
                                    'alignItems': 'center',
                                    'justifyContent': 'center',
                                    'margin': '3% 0 0 0',
                                }),        
                    ]),                    
                    html.Div(className='sub-criteria-confirm-container', style={
                        'width': '100%',
                        'height': '10%',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'alignItems': 'center',
                        'margin': '2.5vh 0 0 0'
                    }, children=[
                        html.Div('Variable objetivo',className='target-variable-text',
                                style={
                                    'width': '100%',
                                    'height': '28%',
                                    'lineHeight': '13vh',
                                    'color': '#FFFFFF',
                                    'fontFamily': "'DM Sans', sans-serif",
                                    'fontSize': '25px',
                                    'fontWeight': '700',
                                    'textAlign': 'center',
                                    'display': 'flex',
                                    'alignItems': 'center',
                                    'justifyContent': 'center',
                                }),
                        html.Div(style={
                            'display': 'flex',
                            'width': '100%',
                            'flexDirection': 'row',
                            'alignItems': 'center',
                            'margin': '2% 0 0 0',
                            'height': '48%'
                        }, children=[
                            html.Div(id='target-variable-container',className='target-variable-container',
                                style={
                                    'position': 'relative',
                                    'width': '70%',
                                    'height': '89%',
                                    'borderRadius': '5px',
                                    'backgroundColor': 'white',
                                    'left': '6%'
                                }),
                            html.Button('OK', className='confirm-button', style={
                                'fontFamily': "'DM Sans', sans-serif",
                                'fontSize': '16px',
                                'fontWeight': '700',
                                'color': 'black',
                                'cursor': 'pointer',
                                'borderRadius': '3px',
                                'borderColor': 'white',
                                'width': '12%',
                                'height': '100%',
                                'backgroundColor': '#11BB52CF',
                                'position': 'relative',
                                'right': '-12%',
                                'top': '9%'
                            }, id="confirm-button-ok", n_clicks=0)
                        ])
                        
                    ])
                    ]
            ),
            # Mapa equipos eventos condiciones
            html.Div(className='graph-container', style={
                'position': 'relative',
                'width': '65.5%',
                'height': '95%',
                'backgroundColor': '#28DB7F',
                'margin': '0 0 0 1.5%',
                'borderRadius': '10px',
                'display': 'flex',
                'flexDirection': 'column',
                'alignItems': 'center'
                }, children=[
                    html.Div('P(X|Y1,Y2,Y3,...,YN)',id='probability_text',className='grap-name-conditional-prob',
                            style={
                                'width': '100%',
                                'height': '10%',
                                'color': '#000000',
                                'fontFamily': "'Poppins, sans-serif",
                                'fontSize': '3vh',
                                'fontWeight': '700',
                                'textAlign': 'center',
                                'display': 'flex',
                                'alignItems': 'center',
                                'justifyContent': 'center',
                            }),
                    html.Div(id='graph-fig-container',className='graph-fig-container', 
                            style={
                                'position': 'relative',
                                'width': '90%',
                                'height': '85%',
                                'backgroundColor': '#FFFFFF',
                                'borderRadius': '10px',
                                'backgroundSize': 'cover',
                                'backgroundPosition': 'center',
                                'backgroundRepeat': 'no-repeat',
                                'display': 'flex',
                                'alignItems': 'center',
                            })
                ]
            )
    ]

# Función para crear el layout
def create_layout():
    return html.Div([
        # Banner superior
        html.Div(className='Banner',children=[
            html.Div(className='Image-User',style={
                'backgroundImage': "url('/assets/images/e0b35f32-93cf-49b5-b63a-248fa22056d1.png')",
                'backgroundSize': 'cover',
                'backgroundPosition': 'center',
                'backgroundRepeat': 'no-repeat',
                'width': '6%',
                'height': '80%',
                'borderRadius': '14px',
                'marginLeft': '7%',
            }),
            html.Div('Hola usuario CHEC', className='Welcome-User',
                     style={
                        'width': '23%',
                        'height': '100%',
                        'borderRadius': '14px',
                        'marginLeft': '2%',
                        'lineHeight': '13vh',
                        'color': '#FFFFFF',
                        'fontFamily': "'DM Sans', sans-serif",
                        'fontSize': '30px',
                        'fontWeight': '700',
                    }),
            html.Div(className='CHEC-Logo',style={
                'backgroundImage': "url('/assets/images/797ea4a7-6ea7-4351-93b9-c76257a788b3.png')",
                'backgroundSize': 'contain',
                'backgroundPosition': 'center',
                'backgroundRepeat': 'no-repeat',
                'width': '16%',
                'height': '80%',
                'borderRadius': '14px',
                'position': 'relative',
                'right': '-45%',
            }),
        ], style={
            'backgroundColor': '#00782b',
            'width': '100%',
            'height': '13.5vh',
            'display': 'flex',
            'flexDirection': 'row',
            'alignItems': 'center',
        }),
        
        # Contenedor inferior
        html.Div([
            # Barra de navegación
            html.Div(className='Nav-Bar', children=[
                html.Button(id='button-maps',className='Maps-Button',style={
                    'width': '100%',
                    'height': '12%',
                    'marginTop': '7vh',
                    'border': '3px solid #068f36',
                    'backgroundColor': '#01471998',
                    'backgroundImage': "url('/assets/images/22ab6d20-fe4b-421e-9ffd-eec28093a1b5.png')",
                    'backgroundSize': '70%',
                    'backgroundPosition': 'center',
                    'backgroundRepeat': 'no-repeat',
                    'cursor': 'pointer',
                }),
                html.Button(className='Graph-Button',style={
                    'width': '100%',
                    'height': '12%',
                    'marginTop': '9vh',
                    'border': '3px solid #068f36',
                    'backgroundColor': '#cdcdcd44',
                    'backgroundImage': "url('/assets/images/7f201cec-29ad-4dc6-ad2c-b331f289fd8a.png')",
                    'backgroundSize': '70%',
                    'backgroundPosition': 'center',
                    'backgroundRepeat': 'no-repeat',
                    
                }),
                html.Button(id='button-chat',className='Chat-Button',style={
                    'width': '100%',
                    'height': '12%',
                    'marginTop': '9vh',
                    'border': '3px solid #068f36',
                    'backgroundColor': '#cdcdcd44',
                    'backgroundImage': "url('/assets/images/ecb71657-f660-4b09-83e9-4f473f3ea97e.png')",
                    'backgroundSize': '70%',
                    'backgroundPosition': 'center',
                    'backgroundRepeat': 'no-repeat',
                    'cursor': 'pointer',
                }),
                html.Button(id='button-tab-net',className='TabNet-Button',style={
                    'width': '100%',
                    'height': '12%',
                    'marginTop': '9vh',
                    'border': '3px solid #068f36',
                    'backgroundColor': '#cdcdcd44',
                    'backgroundImage': "url('/assets/images/stats-graph-svgrepo-com.svg')",
                    'backgroundSize': '70%',
                    'backgroundPosition': 'center',
                    'backgroundRepeat': 'no-repeat',
                    'cursor': 'pointer',
                }),
                dcc.Location(id='url-graphs', refresh=True)
            ], style={
                'backgroundColor': '#00782b',
                'width': '5.83%',
                'height': '86.5vh',
                'display': 'flex',
                'flexDirection': 'column',
            }),
            
            # Espacio de trabajo con children dinámico
            html.Div(className='Work-Space',children=work_space(), style={
                'backgroundColor': '#FFFFFF',
                'width': '94.17%',
                'height': '86.5vh',
                'display': 'flex',
                'flexDirection': 'row',
                'alignItems': 'center',
            })
        ], style={
            'display': 'flex',
            'flex': '1',
        })
    ], style={
        'height': '100vh',
        'margin': 0,
        'display': 'flex',
        'flexDirection': 'column'
    })