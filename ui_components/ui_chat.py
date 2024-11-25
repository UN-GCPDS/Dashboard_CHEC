import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Output, Input, callback_context
from dash.dependencies import Input, Output, State, ALL

from functions.utils import load_previous_conversations, save_conversations, conversation, load_structured_data, update_documents_procces

# Función que genera el contenido del espacio de trabajo
def work_space():

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
    ]

    PROCESOS_UPLOAD = [proceso for proceso in PROCESOS if proceso["value"] != "general"]

    # Puede ser un solo elemento o una lista de elementos
    return [

        dbc.Col(id='chats-history', width=3, children=[
            dbc.Card(
                dbc.CardBody([
                    html.H5(
                        "Actualizar Documento", 
                        style={
                            'color': '#00782B',
                            'fontWeight': 'bold',
                            'textAlign': 'center',
                            'marginBottom': '10px',
                            'fontSize': '1.1em',
                            'fontFamily': "'DM Sans', sans-serif"
                        }
                    ),
                    html.Div([
                        html.Label(
                            "Selecciona el Documento a Actualizar:", 
                            style={'color': '#00782B', 'fontWeight': 'bold', 'fontFamily': "'DM Sans', sans-serif"}
                        ),
                        dcc.Dropdown(
                            id='upload-proceso-select',
                            options=PROCESOS_UPLOAD,
                            placeholder="Selecciona un Proceso",
                            style={
                                'width': '100%',
                                'borderRadius': '10px',
                                'borderColor': '#80A022',
                                'backgroundColor': 'white',
                                'color': '#00782B',
                                'padding': '5px',
                                'fontFamily': "'Poppins', sans-serif"
                            }
                        )
                    ], style={'margin': '3vh 0 3vh 0'}),
                    dcc.Upload(
                        id='upload-pdf',
                        children=html.Div([
                            'Sube el Documento',
                            html.A('', style={'color': '#00782B',})
                        ], style={'fontFamily': "'DM Sans', sans-serif"}),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '10px',
                            'backgroundColor': '#E6F4EA',
                            'color': '#00782B',
                            'textAlign': 'center',
                            'cursor': 'pointer'
                        },
                        accept='.pdf'
                    ),
                    html.Div(id='upload-status', style={'marginTop': '8px'})
                ]),
                style={
                    'backgroundColor': '#F0F6E7',
                    'padding': '15px',
                    'marginBottom': '-11px'
                }
            ),
            html.H2(
                "Historial de Chats", 
                style={'color': '#00782B', 'fontFamily': "'DM Sans', sans-serif", 'textAlign': 'center', 'margin': '0% 0% 0% 0%'}
            ),
            html.Hr(),
            dbc.Button(
                "Nuevo Chat", 
                id="nuevo-chat", 
                style={
                    'backgroundColor': '#80A022',
                    'color': 'white',
                    'border': 'none',
                    'width': '100%'
                },
                className="mb-3"
            ),
            html.Div(
                id='lista-chats', 
                children=[]
            )
        ], style={
            'height': '99.9%',
            'width': '25%',
            'overflowY': 'auto',
            'backgroundColor': '#F0F6E7',
            'transition': 'height 0.5s ease'
        }),
            
        html.Div(id='chat-window',style={
            'display': 'flex',
            'flexDirection': 'row',
            'width': '75%',
            'height': '100%',
            'backgroundColor': '#f0f6e7',
            'transition': 'height 0.5s ease'
        },children=[
            html.Button(id='toggle-button',className='Chat-Button',style={
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
                }),
            dcc.Store(id='toggle-state', data=False),  # False -> Div colapsado, True -> Div expandido
            dbc.Col(width=9, children=[
                
                html.Div(
                    id='ventana-chat', 
                    style={
                        'height': '82%',
                        'overflowY': 'scroll',
                        'padding': '10px',
                        'backgroundColor': '#f0f6e7',
                        'display': 'flex',
                        'flex-direction': 'column',
                    }
                ),

                dbc.Row([
                    dbc.Col(width=3, children=[
                        dcc.Dropdown(
                            id='model-select',
                            options=[
                                {'label': 'GPT', 'value': 'gpt'},
                                {'label': 'Llama 3.1', 'value': 'llama3.1'},
                                {'label': 'Llama 3.2', 'value': 'llama3.2'}
                            ],
                            value='gpt',
                            clearable=False,
                            style={'borderRadius': '5px'}
                        )
                    ], style={
                        'marginBottom': '8px',
                        'width': '20%',
                        }),
                    dbc.Col(width=3, children=[
                        dcc.Dropdown(
                            id='proceso-select',
                            options=PROCESOS,
                            value='general',
                            clearable=False,
                            style={'borderRadius': '5px'}
                        )
                    ], style={
                        'margin': '0 0 8px 0',
                        'width': '32%'
                        })
                ], style={
                    'display': 'flex',
                    'flexDirection': 'row',
                    'margin': '2vh 0 0 2%',
                }),

                dbc.InputGroup([
                    dbc.Input(
                        id='entrada-usuario',
                        placeholder='Escribe tu pregunta...',
                        type='text',
                        style={
                            'borderColor': '#80A022',
                            'width': '87%',
                            'height': '70%',
                            'border-radius': '10px',
                            'margin': '0 0 0 2%'
                            }
                    ),
                    dbc.Button(
                        "Enviar",
                        id='enviar-btn',
                        n_clicks=0,
                        style={
                            'backgroundColor': '#80A022',
                            'color': 'white',
                            'borderRadius': '10px',
                            'width': 'auto',
                            'height': '42%',
                            'margin': '0 0 0 2%',
                        }
                    )
                    ], style={
                    'width': '100%', 
                    'height': '16%'})
                ], style={
                'backgroundColor': '#f0f6e7',
                'display': 'flex',
                'flex-direction': 'column',
                'width': '97%',
                'height': '99.9%',
                'overflow': 'visible'
            }),
            dcc.Store(id='chat-data', data=load_previous_conversations())
        ])
            
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
                    'marginTop': '12vh',
                    'border': '3px solid #068f36',
                    'backgroundColor': '#01471998',
                    'backgroundImage': "url('/assets/images/22ab6d20-fe4b-421e-9ffd-eec28093a1b5.png')",
                    'backgroundSize': '70%',
                    'backgroundPosition': 'center',
                    'backgroundRepeat': 'no-repeat',
                    'cursor': 'pointer',
                }),
                html.Button(id='button-graphs',className='Graph-Button',style={
                    'width': '100%',
                    'height': '12%',
                    'marginTop': '12vh',
                    'border': '3px solid #068f36',
                    'backgroundColor': '#cdcdcd44',
                    'backgroundImage': "url('/assets/images/7f201cec-29ad-4dc6-ad2c-b331f289fd8a.png')",
                    'backgroundSize': '70%',
                    'backgroundPosition': 'center',
                    'backgroundRepeat': 'no-repeat',
                    'cursor': 'pointer',
                }),
                html.Button(className='Chat-Button',style={
                    'width': '100%',
                    'height': '12%',
                    'marginTop': '12vh',
                    'border': '3px solid #068f36',
                    'backgroundColor': '#cdcdcd44',
                    'backgroundImage': "url('/assets/images/ecb71657-f660-4b09-83e9-4f473f3ea97e.png')",
                    'backgroundSize': '70%',
                    'backgroundPosition': 'center',
                    'backgroundRepeat': 'no-repeat',
                }),
                dcc.Location(id='url-chat', refresh=True)
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