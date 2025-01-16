from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

# Función que genera el contenido del espacio de trabajo
def work_space():
    # Puede ser un solo elemento o una lista de elementos
    return [
            # Selector de fecha
            html.Div(
                className="selector-date",
                style={
                    'width': '98%',
                    'height': '10vh',
                    'margin': '21px 0 0 0',
                    'background': 'rgba(0, 120, 43, 0.76)',
                    'zIndex': '8',
                    'borderRadius': '9px',
                    'overflow': 'visible',
                    'display': 'flex',
                    'flexDirection': 'row',
                    'align-items': 'center',

                },
                children=[
                    html.Div(className="date-icon",style={
                    'backgroundImage': "url('/assets/images/2529e910-74c1-45f9-acd7-0e598c74583b.png')",
                    'backgroundSize': 'contain',
                    'backgroundPosition': 'center',
                    'backgroundRepeat': 'no-repeat',
                    'width': '6%',
                    'height': '50%',
                    'borderRadius': '14px',
                    'marginLeft': '0%',
                    }),
                    html.Div('SELECCIONAR FECHA', className="text-date", style={
                        'fontFamily': "'DM Sans', sans-serif",
                        'fontSize': '16px',
                        'fontWeight': '700',
                        'color': 'white',
                        'width': '12%'
                    }),
                    html.Div(id='date-container', className="date-container", style={
                        'backgroundColor': 'white',
                        'width': '12%',
                        'borderColor': 'white',
                        'borderRadius': '9px',
                        'height': '6vh',
                        'margin': '0 0 0 1%',
                        'display': 'flex',
                        'align-items': 'center'
                    }),
                    html.Div('SELECCIONAR MUNICIPIO', className="text-municipio", style={
                        'fontFamily': "'DM Sans', sans-serif",
                        'fontSize': '16px',
                        'fontWeight': '700',
                        'color': 'white',
                        'margin': '0 0 0 1%',
                        'width': '14%'
                    }),
                    html.Div(id='municipio-container',className="municipio-container", style={
                        'backgroundColor': 'white',
                        'width': '18%',
                        'borderColor': 'white',
                        'borderRadius': '9px',
                        'height': '6vh',
                        'margin': '0 0 0 1%',
                        'display': 'flex',
                        'align-items': 'center'
                    }),
                    html.Div('TIPO CONDICIÓN', className="text-cond-env", style={
                        'fontFamily': "'DM Sans', sans-serif",
                        'fontSize': '16px',
                        'fontWeight': '700',
                        'color': 'white',
                        'margin': '0 0 0 1%',
                        'width': '12%'
                    }),
                    html.Div(id='cond-env-container',className="cond-env-container", style={
                        'backgroundColor': 'white',
                        'width': '12%',
                        'borderColor': 'white',
                        'borderRadius': '9px',
                        'height': '6vh',
                        'margin': '0 0 0 1%',
                        'display': 'flex',
                        'align-items': 'center'
                    }),
                    html.Button('OK', className='confirm-button', style={
                        'position': 'absolute',
                        'fontFamily': "'DM Sans', sans-serif",
                        'fontSize': '16px',
                        'fontWeight': '700',
                        'color': 'black',
                        'cursor': 'pointer',
                        'borderRadius': '3px',
                        'borderColor': 'white',
                        'right': '2.5%',
                        'width': '3.5%',
                        'height': '5vh',
                        'backgroundColor': '#11BB52CF',
                    }, id="confirm-button", n_clicks=0)  # Para callbacks de Dash
                ]
            ),
            # Mapa equipos eventos condiciones
            html.Div(
                className="mapa-equipos-eventos-condiciones",
                style={
                    'width': '98%',
                    'height': '70vh',
                    'margin': '18px 0 0 0',
                    'background': 'rgba(45, 154, 35, 0.8)',
                    'zIndex': '1',
                    'borderRadius': '9px',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'justify-content': 'center'
                },
                children=[
                    html.Div(className="map-banner",
                            style={
                                'display': 'flex',
                                'flexDirection': 'row',
                                'position': 'relative',
                                'height': '6vh',
                                'align-items': 'center',
                                'margin': '-1vh 0 0 0',
                            }
                            ,children=[
                                html.Div(className="map-icon",style={
                                    'backgroundImage': "url('/assets/images/6b0cfc3d-739d-432b-a281-b5f18100d3bc.png')",
                                    'backgroundSize': 'contain',
                                    'backgroundPosition': 'center',
                                    'backgroundRepeat': 'no-repeat',
                                    'width': '6%',
                                    'height': '80%',
                                    'borderRadius': '14px',
                                    'marginLeft': '0%',
                                }),
                                html.Div('Mapa de equipos, eventos y condiciones ambientales', className="text-date", style={
                                    'fontFamily': "'DM Sans', sans-serif",
                                    'fontSize': '20px',
                                    'fontWeight': '700',
                                    'color': 'white',
                                })]
                            ),
                    html.Div(id='map-container',className="map-container", 
                            style={
                                'backgroundColor': 'white',
                                'width': '97.6%',
                                'borderColor': 'white',
                                'borderRadius': '9px',
                                'height': '60vh',
                                'margin': '1vh 0px 0px 1.2%',
                                
                            }),
                    html.Div(id='select-evento', children=[]),
                    html.Div(id='recomendacion-button', children=[]),
                    ], 
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
                html.Button(className='Maps-Button',style={
                    'width': '100%',
                    'height': '12%',
                    'marginTop': '7vh',
                    'border': '3px solid #068f36',
                    'backgroundColor': '#01471998',
                    'backgroundImage': "url('/assets/images/22ab6d20-fe4b-421e-9ffd-eec28093a1b5.png')",
                    'backgroundSize': '70%',
                    'backgroundPosition': 'center',
                    'backgroundRepeat': 'no-repeat',
                }),
                html.Button(id='button-graphs',className='Graph-Button',style={
                    'width': '100%',
                    'height': '12%',
                    'marginTop': '9vh',
                    'border': '3px solid #068f36',
                    'backgroundColor': '#cdcdcd44',
                    'backgroundImage': "url('/assets/images/7f201cec-29ad-4dc6-ad2c-b331f289fd8a.png')",
                    'backgroundSize': '70%',
                    'backgroundPosition': 'center',
                    'backgroundRepeat': 'no-repeat',
                    'cursor': 'pointer',
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
                dcc.Location(id='url-maps', refresh=True)
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
                'flexDirection': 'column',
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





