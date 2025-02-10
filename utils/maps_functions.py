import io
import math
import folium
import html
import pandas as pd
import numpy as np
from datetime import datetime
from folium.plugins import HeatMap
import networkx as nx
from scipy.spatial import ConvexHull
from shapely.geometry import LineString, Point

import json
import torch
import webbrowser
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pyvis.network import Network
from scipy.special import softmax
from unidecode import unidecode
from scipy.spatial.distance import cdist, squareform
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.neighbors import NearestNeighbors
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
from pytorch_tabnet.augmentations import RegressionSMOTE
import os
from pathlib import Path
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm, colors, pyplot as plt

import sys
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

def safe_write_html(grafo, filename):
    """
    Escribe el HTML del grafo de manera segura manejando problemas de codificación
    """
    try:
        # Intentar escribir con diferentes estrategias de codificación
        with open(filename, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(grafo.html)
    except Exception as e:
        # Escribir sin especificar codificación
        with open(filename, 'w') as f:
            f.write(grafo.html)
        print(f"Advertencia: Error al escribir {filename}. Detalles: {e}")

# Función auxiliar para etiquetas
def get_labels(x: pd.Series) -> pd.Series:
    labels, _ = pd.factorize(x)
    return pd.Series(labels, name=x.name, index=x.index)

# Definición de funciones personalizadas de pérdida
def my_mse_loss_fn(y_pred, y_true):
    mse_loss = (y_true - y_pred) ** 2
    return torch.mean(mse_loss)

def my_rmse_loss_fn(y_pred, y_true):
    mse_loss = (y_true - y_pred) ** 2
    mean_mse_loss = torch.mean(mse_loss)
    rmse_loss = torch.sqrt(mean_mse_loss)
    return rmse_loss

def my_mae_loss_fn(y_pred, y_true):
    mae_loss = torch.abs(y_true - y_pred)
    return torch.mean(mae_loss)

def my_mape_loss_fn(y_pred, y_true):
    mape_loss = torch.abs((y_true - y_pred) / y_true) * 100
    return torch.mean(mape_loss)

def my_r2_score_fn(y_pred, y_true):
    total_variance = torch.var(y_true, unbiased=False)
    unexplained_variance = torch.mean((y_true - y_pred) ** 2)
    r2_score = 1 - (unexplained_variance / total_variance)
    return 1-r2_score

# Clase personalizada para TabNetRegressor
class CustomTabNetRegressor(TabNetRegressor):
    def __init__(self, *args, **kwargs):
        super(CustomTabNetRegressor, self).__init__(*args, **kwargs)

    def forward(self, X):
        output, M_loss = self.network(X)
        output = torch.relu(output)
        return output, M_loss

    def predict(self, X):
        device = next(self.network.parameters()).device
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(device)
        with torch.no_grad():
            output, _ = self.forward(X)
        return output.cpu().numpy()

# Parámetros
par = {
    'n_d': 144, 'n_a': 144, 'n_steps': 10, 'gamma': 94.66997047890686,
    'lambda_sparse': 2.8731055681649033e-11, 'batch_size': 4096,
    'mask_type': 'entmax', 'emb': 46, 'momentum': 0.023136657722718557,
    'learning_rate': 0.03017683806097458, 'weight_decay': 4.1323153592424204e-05,
    'scheduler_gamma': 0.44928231250804757, 'step_size': 15,
    'virtual_batch_size': 1024, 'optimizer_type': 'rmsprop',
    'p': 0.9806570564809924
}

reglas_equipo = {
    "apoyo": {
        "permitidos": {
            "MATERIAL", "LONG_APOYO", "TIERRA_PIE", "VIENTO",
            "ÍNDICE_RAYOS", "PRECIPITACIÓN", "RADIACIÓN_UV", "TEMPERATURA"
        },
        "keywords": {"temp", "solar_rad", "uv", "wind_gust_spd", "wind_spd"}
    },
    "switch": {
        "permitidos": {
            "KV", "PHASES", "STATE", "VELOCIDAD_VIENTO",
            "TEMPERATURA_AMBIENTE", "PRECIPITACIÓN",
            "HUMEDAD_RELATIVA", "RAYOS"
        },
        "keywords": {"rh", "temp", "wind_gust_spd", "wind_spd"}
    },
    "tramo_red": {
        "permitidos": {
            "KVNOM", "MATERIALCONDUCTOR", "CALIBRECONDUCTOR",
            "GUARDACONDUCTOR", "VELOCIDAD_VIENTO", "TEMPERATURA",
            "HUMEDAD_RELATIVA", "PRECIPITACIÓN"
        },
        "keywords": {"wind_gust_spd", "wind_spd", "rh"}
    },
    "transformador": {
        "permitidos": {
            "KVA", "KV1", "IMPEDANCE", "Temperatura",
            "Humedad", "Aceite Aislante"
        },
        "keywords": {"temp", "rh"}
    }
}

def procesar_json(entrada_json):
    resultado = {}

    for codigo, datos in entrada_json.items():
        tipo_equipo = datos["Tipo_de_equipo"]
        top_10 = datos["top_5"]

        if tipo_equipo in reglas_equipo:
            reglas = reglas_equipo[tipo_equipo]
            nuevo_top = {}

            for clave, valor in top_10.items():
                # Verificar si la clave está permitida o contiene una keyword
                if clave in reglas["permitidos"] or any(kw in clave for kw in reglas["keywords"]):
                    nuevo_top[clave] = valor

            # Si hay más de 5 elementos, conservar solo los primeros 5
            nuevo_top = dict(list(nuevo_top.items())[:5])

            resultado[codigo] = {
                "Tipo_de_equipo": tipo_equipo,
                "top_5": nuevo_top
            }
        else:
            # Si el tipo de equipo no tiene reglas, se guarda sin cambios
            resultado[codigo] = datos

    return resultado

def graficar_grafo_interactivo_2(Z, nombres_columnas, num=0, height=400, width=70, iteraciones=20, vector=None):
    """
    Función para graficar un grafo interactivo con Pyvis y mostrar una barra de colores.
    """
    # Calcular la matriz de similitud
    D_ = cdist(Z.T, Z.T)  # Cambiar Z a Z.T si los datos deben ser transpuestos
    sig_ = 0.05 * np.median(squareform(D_))
    eps = 1e-10  # Pequeño valor para evitar división por cero
    A = np.exp(-D_**2 / (sig_**2 + eps))

    # Crear el grafo y eliminar aristas con pesos bajos
    G = nx.from_numpy_array(A - np.eye(A.shape[0]))
    edges_to_remove = [(u, v) for u, v, data in G.edges(data=True) if data['weight'] < 1e-8]
    G.remove_edges_from(edges_to_remove)

    # Calcular colores de nodos usando el vector si se proporciona
    if vector is not None:
        mean_values = vector
    else:
        mean_values = np.mean(Z, axis=0)

    # Normalizar los valores
    mean_values = (mean_values - np.min(mean_values)) / (np.max(mean_values) - np.min(mean_values))
    norm = colors.Normalize(vmin=np.min(mean_values), vmax=np.max(mean_values))
    cmap = cm.get_cmap('viridis')  # Cambiar el colormap si lo deseas

    # Asignar colores a los nodos
    node_colors = [colors.to_hex(cmap(norm(value))) for value in mean_values]

    # Crear el grafo interactivo con Pyvis
    net = Network(notebook=True, height=f"{height}px", width=f"{width}%", bgcolor="#FFFFFF", font_color="black", cdn_resources="in_line")

    for i, node in enumerate(G.nodes):
        # Convertir nombres de columnas a cadenas ASCII seguros
        safe_label = unidecode(str(nombres_columnas[node]))
        safe_title = unidecode(f"{nombres_columnas[node]}: {mean_values[i]:.2f}")

        net.add_node(
            node,
            label=safe_label,  # Nombre del nodo
            color=node_colors[i],         # Color del nodo basado en mean_values
            title=safe_title,  # Tooltip al pasar el ratón
            font={'size': 20}             # Tamaño del texto
        )

    # Agregar aristas al grafo
    for edge in G.edges:
        net.add_edge(edge[0], edge[1])

    # Configurar las opciones de simulación física con iteraciones dinámicas
    net.set_options(f"""
    var options = {{
        "physics": {{
            "enabled": true,
            "stabilization": {{
                "enabled": true,
                "iterations": {iteraciones}
            }},
            "solver": "forceAtlas2Based"
        }}
    }}
    """)
    
    return net


def process_dataframe(redmt, df, label_encoders, df1, ind,tip,s,scolumns, NUMERIC_COLUMNS, max_values):
    """
    Procesa un DataFrame `redmt` usando coordenadas y LabelEncoders, y devuelve un DataFrame resultante.

    Args:
        redmt (pd.DataFrame): DataFrame base con información de referencia.
        df (pd.DataFrame): DataFrame con coordenadas para encontrar vecinos cercanos.
        label_encoders (dict): Diccionario con `LabelEncoder` para columnas categóricas.
        df1 (pd.DataFrame): DataFrame con la fila de interés.
        ind (int): Índice de la fila de interés en `df1`.

    Returns:
        pd.DataFrame: DataFrame resultante con los datos procesados.
    """
    # Convertir las columnas de fecha
    df1['inicio'] = pd.to_datetime(df1['inicio'])
    df1['FECHA_C'] = df1['inicio'].dt.to_period('M')

    # Seleccionar la fila de interés de `df1` (por índice)
    row_of_interest = df1.loc[[ind]].copy()
    #print('1',row_of_interest[['LATITUD','LONGITUD']].values)
    #row_of_interest[scolumns]=np.nan
    # Vaciar los valores de las columnas en `scolumns`
    #for col in scolumns:
    #    if col in row_of_interest.columns:
    #        row_of_interest[col] = np.nan

    # Extraer las listas de coordenadas y equipos desde la fila de interés
    if s==0:
        #aux = eval(row_of_interest.loc[ind, 'TRAMOS_AGUAS_ABAJO'])
        aux=  list(eval(row_of_interest.loc[ind,'TRAMOS_AGUAS_ABAJO_CODES']))
    else:
        aux = eval(row_of_interest.loc[ind, 'EQUIPOS_PUNTOS'])


    # DataFrame para almacenar las nuevas filas
    new_rows = []
    # Iterar sobre cada elemento de `aux` para filtrar y duplicar
    for i in aux:
        # Filtrar `redmt` según las condiciones dadas
        if s==0:
          filtered_row = redmt[
              (redmt['FECHA_C'] == row_of_interest.loc[ind, 'FECHA_C']) &
              (redmt['equipo_ope']== i)
          ]
        else:
          filtered_row = redmt[
              (redmt['FECHA_C'] == row_of_interest.loc[ind, 'FECHA_C']) &
              (redmt['LATITUD'] == i[0]) &
              (redmt['LONGITUD'] == i[1])]
        #print('2',filtered_row[['LATITUD','LONGITUD']].values)
        # Si hay filas que cumplen la condición, reemplazar columnas en la fila de interés
        if not filtered_row.empty:
            for _, row in filtered_row.iterrows():
                #print(3,redmt.columns)
                # Crear una copia de la fila de interés y reemplazar las columnas correspondientes
                temp_row = row_of_interest.copy()
                temp_row[redmt.columns] = row.values  # Reemplaza las columnas de redmt
                #temp_row['LATITUD'] = np.float64(i[0])  # Asegura precisión en la asignación
                #temp_row['LONGITUD'] = np.float64(i[1])
                new_rows.append(temp_row)
    if not new_rows:
        # Retornar un DataFrame vacío con las columnas esperadas
        aux1=pd.DataFrame(columns=df1.columns)
        aux1.drop(['inicio_evento', 'h0-solar_rad', 'h0-uv', 'h1-solar_rad', 'h1-uv', 'h2-solar_rad', 'h2-uv', 'h3-solar_rad', 'h3-uv',
            'h4-solar_rad', 'h4-uv', 'h5-solar_rad', 'h5-uv', 'h19-solar_rad', 'h19-uv', 'h20-solar_rad', 'h20-uv',
            'h21-solar_rad', 'h21-uv', 'h22-solar_rad', 'h22-uv', 'h23-solar_rad', 'h23-uv', 'evento', 'fin', 'inicio',
            'cnt_usus', 'DEP', 'MUN', 'FECHA', 'NIVEL_C', 'VALOR_C', 'TRAMOS_AGUAS_ABAJO', 'EQUIPOS_PUNTOS',
            'PUNTOS_POLIGONO', 'LONGITUD2', 'LATITUD2', 'FECHA_C','TRAMOS_AGUAS_ABAJO_CODES','ORDER_'],
           inplace=True, axis=1)
        aux1.drop(['SAIFI', 'SAIDI', 'duracion_h'], axis=1, inplace=True)
        return pd.DataFrame(columns=scolumns).values,aux1
    # Concatenar todas las nuevas filas generadas
    result_df = pd.concat(new_rows, ignore_index=True)
    result_df.drop_duplicates(inplace=True)
    result_df['LATITUD'] = result_df['LATITUD'].astype('float64')
    result_df['LONGITUD'] = result_df['LONGITUD'].astype('float64')
    result_df1 =result_df.copy()
    result_df1['LATITUD'] = result_df1['LATITUD'].astype('float64')
    result_df1['LONGITUD'] = result_df1['LONGITUD'].astype('float64')

    # Codificar las columnas categóricas usando los LabelEncoder definidos en `label_encoders`
    for col, le in label_encoders.items():
      if col in result_df.columns:  # Verificar que la columna exista en `result_df`
        if col in redmt.columns:  # Si la columna pertenece a redmt
            result_df[col] = result_df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else np.nan
            )
        else:  # Si no pertenece a redmt
            result_df[col] = result_df[col].fillna("no aplica")  # Rellenar NaN con "no aplica"
            result_df[col] = result_df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )


    # Reemplazar valores NaN en columnas categóricas usando el valor más cercano
    categorical_columns = redmt.select_dtypes(include=['object', 'category']).columns

    # Preparar las coordenadas (LATITUD y LONGITUD) de `df`
    df_coords = df[['LATITUD', 'LONGITUD']].dropna()

    # Modelo de vecinos más cercanos
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(df_coords)

    # Recorrer las columnas categóricas de result_df
    for col in result_df.columns:
        if col in redmt.columns:  # Verificar si la columna pertenece a redmt
            nan_indices = result_df[result_df[col].isna()].index  # Índices con NaN en la columna
            for idx in nan_indices:
                # Coordenadas de la fila con NaN
                query_coords = result_df.loc[idx, ['LATITUD', 'LONGITUD']].values.reshape(1, -1)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="X does not have valid feature names, but NearestNeighbors was fitted with feature names")
                    distance, neighbor_idx = nbrs.kneighbors(query_coords)
                closest_idx = df_coords.iloc[neighbor_idx[0][0]].name
                # Reemplazar el valor NaN con el valor del vecino más cercano
                result_df.at[idx, col] = df.at[closest_idx, col]
        #else:
            #result_df[col].astype(str).fillna("no aplica",inplace=True)
    for col in NUMERIC_COLUMNS:
        max_value = max_values[col]
        # Rellenar valores y ajustar el tipo de datos
        result_df[col] = result_df[col].fillna(-10 * max_value).astype('float64')

    result_df['tipo_equi_ope'] = tip
    result_df.drop(['inicio_evento', 'h0-solar_rad', 'h0-uv', 'h1-solar_rad', 'h1-uv', 'h2-solar_rad', 'h2-uv', 'h3-solar_rad', 'h3-uv',
            'h4-solar_rad', 'h4-uv', 'h5-solar_rad', 'h5-uv', 'h19-solar_rad', 'h19-uv', 'h20-solar_rad', 'h20-uv',
            'h21-solar_rad', 'h21-uv', 'h22-solar_rad', 'h22-uv', 'h23-solar_rad', 'h23-uv', 'evento', 'fin', 'inicio',
            'cnt_usus', 'DEP', 'MUN', 'FECHA', 'NIVEL_C', 'VALOR_C', 'TRAMOS_AGUAS_ABAJO', 'EQUIPOS_PUNTOS',
            'PUNTOS_POLIGONO', 'LONGITUD2', 'LATITUD2', 'FECHA_C','TRAMOS_AGUAS_ABAJO_CODES','ORDER_'],
           inplace=True, axis=1)
    result_df.drop(['SAIFI', 'SAIDI', 'duracion_h'], axis=1, inplace=True)
    result_df1.drop(['inicio_evento', 'h0-solar_rad', 'h0-uv', 'h1-solar_rad', 'h1-uv', 'h2-solar_rad', 'h2-uv', 'h3-solar_rad', 'h3-uv',
            'h4-solar_rad', 'h4-uv', 'h5-solar_rad', 'h5-uv', 'h19-solar_rad', 'h19-uv', 'h20-solar_rad', 'h20-uv',
            'h21-solar_rad', 'h21-uv', 'h22-solar_rad', 'h22-uv', 'h23-solar_rad', 'h23-uv', 'evento', 'fin', 'inicio',
            'cnt_usus', 'DEP', 'MUN', 'FECHA', 'NIVEL_C', 'VALOR_C', 'TRAMOS_AGUAS_ABAJO', 'EQUIPOS_PUNTOS',
            'PUNTOS_POLIGONO', 'LONGITUD2', 'LATITUD2', 'FECHA_C','TRAMOS_AGUAS_ABAJO_CODES','ORDER_'],
           inplace=True, axis=1)
    result_df1.drop(['SAIFI', 'SAIDI', 'duracion_h'], axis=1, inplace=True)
    result_df1.drop_duplicates(inplace=True)
    result_df.drop_duplicates(inplace=True)
    return result_df.values,result_df1

def enumerate_repeated_from_startup(lista):
    # Diccionario para contar ocurrencias totales
    conteo_total = {}
    for elemento in lista:
        conteo_total[elemento] = conteo_total.get(elemento, 0) + 1
    
    # Diccionario para seguir el conteo actual
    conteo_actual = {}
    resultado = []
    
    for elemento in lista:
        if conteo_total[elemento] > 1:
            # Si el elemento aparece más de una vez
            conteo_actual[elemento] = conteo_actual.get(elemento, 0) + 1
            resultado.append(f"{elemento}-{conteo_actual[elemento]}")
        else:
            # Si el elemento aparece solo una vez
            resultado.append(elemento)
            
    return resultado


def select_data(año,mes,mun,trafos,apoyos,switches,redmt,super_eventos, descargas, vegetacion, df1):
    trafos_seleccionado = trafos.loc[(trafos['FECHA'].dt.year == año) & (trafos['FECHA'].dt.month == mes) & (trafos['MUN'] == mun)]
    apoyos_seleccionado = apoyos.loc[(apoyos['FECHA'].dt.year == año) & (apoyos['FECHA'].dt.month == mes) & (apoyos['MUN'] == mun)]
    redmt_seleccionado = redmt.loc[(redmt['FECHA'].dt.year == año) & (redmt['FECHA'].dt.month == mes) & (redmt['MUN'] == mun)]
    switches_seleccionado = switches.loc[(switches['FECHA'].dt.year == año) & (switches['FECHA'].dt.month == mes) & (switches['MUN'] == mun)]

    super_eventos_seleccionado = super_eventos.loc[(super_eventos['inicio'].dt.year == año) & (super_eventos['inicio'].dt.month == mes) & (super_eventos['MUN'] == mun)]
    # Crear una lista de DataFrames para cada día del mes, del 1 al 31
    super_eventos_seleccionado_1 = []
    for dia in range(1, 32):  # Del día 1 al 31
        df_dia = super_eventos_seleccionado[super_eventos_seleccionado['inicio'].dt.day == dia]       
        # Añadir el DataFrame a la lista para el seguimiento
        super_eventos_seleccionado_1.append(df_dia)

    descargas_seleccionado = descargas.loc[(descargas['FECHA'].dt.year == año) & (descargas['FECHA'].dt.month == mes) & (descargas['MUN'] == mun)]
    # Crear una lista de DataFrames para cada día del mes, del 1 al 31
    descargas_seleccionado_1 = []
    for dia in range(1, 32):  # Del día 1 al 31
        df_dia = descargas_seleccionado[descargas_seleccionado['FECHA'].dt.day == dia]       
        # Añadir el DataFrame a la lista para el seguimiento
        descargas_seleccionado_1.append(df_dia)

    vegetacion_seleccionado = vegetacion.loc[(vegetacion['FECHA'].dt.year == año) & (vegetacion['FECHA'].dt.month == mes) & (vegetacion['MUN'] == mun)]
    # Crear una lista de DataFrames para cada día del mes, del 1 al 31
    vegetacion_seleccionado_1 = []
    for dia in range(1, 32):  # Del día 1 al 31
        df_dia = vegetacion_seleccionado[vegetacion_seleccionado['FECHA'].dt.day == dia]       
        # Añadir el DataFrame a la lista para el seguimiento
        vegetacion_seleccionado_1.append(df_dia)
    
    df1_seleccionado = df1.loc[(df1['inicio'].dt.year == año) & (df1['inicio'].dt.month == mes) & (df1['MUN'] == mun)]
    df1_seleccionado_1 = []
    for dia in range(1, 32):  # Del día 1 al 31
        df_dia = df1_seleccionado[df1_seleccionado['inicio'].dt.day == dia]       
        # Añadir el DataFrame a la lista para el seguimiento
        df1_seleccionado_1.append(df_dia)


    return trafos_seleccionado, apoyos_seleccionado, switches_seleccionado, redmt_seleccionado, super_eventos_seleccionado_1, descargas_seleccionado_1, vegetacion_seleccionado_1, df1_seleccionado_1

def load_data():

    trafos = pd.read_pickle(os.path.join("..", "data", "TRAFOS.pkl"))
    trafos['FECHA']=pd.to_datetime(trafos['FECHA'])
    trafos['FECHA_C']=trafos['FECHA'].dt.to_period('M')
    trafos.rename(columns={'CODE':'equipo_ope'}, inplace=True)

    apoyos = pd.read_pickle(os.path.join("..", "data", "APOYOS.pkl"))
    apoyos['FECHA']=pd.to_datetime(apoyos['FECHA'])
    apoyos['FECHA_C']=apoyos['FECHA'].dt.to_period('M')
    apoyos.rename(columns={'CODE':'equipo_ope'}, inplace=True)

    redmt = pd.read_pickle(os.path.join("..", "data", "REDMT.pkl"))
    redmt['FECHA']=pd.to_datetime(redmt['FECHA'])
    redmt['FECHA_C']=redmt['FECHA'].dt.to_period('M')
    redmt.rename(columns={'CODE':'equipo_ope'}, inplace=True)

    switches = pd.read_pickle(os.path.join("..", "data", "SWITCHES.pkl"))
    switches['FECHA']=pd.to_datetime(switches['FECHA'])
    switches['FECHA_C']=switches['FECHA'].dt.to_period('M')
    switches.rename(columns={'CODE':'equipo_ope'}, inplace=True)

    super_eventos = pd.read_pickle(os.path.join("..", "data", "SuperEventos_Criticidad_AguasAbajo_CODEs.pkl"))

    descargas = pd.read_pickle(os.path.join("..", "data", "Rayos.pkl"))

    vegetacion = pd.read_pickle(os.path.join("..", "data", "Vegetacion.pkl"))

    Xdata = super_eventos.copy()
    Xdata = Xdata[Xdata['duracion_h'] <= 100]

    # Extraer variables objetivo
    Dur_h = Xdata['duracion_h'].values
    SAIDI = Xdata['SAIDI'].values
    df1=Xdata.copy()
    # Eliminar columnas no utilizadas
    Xdata.drop(['inicio_evento', 'h0-solar_rad', 'h0-uv', 'h1-solar_rad', 'h1-uv', 'h2-solar_rad', 'h2-uv', 'h3-solar_rad', 'h3-uv',
                'h4-solar_rad', 'h4-uv', 'h5-solar_rad', 'h5-uv', 'h19-solar_rad', 'h19-uv', 'h20-solar_rad', 'h20-uv',
                'h21-solar_rad', 'h21-uv', 'h22-solar_rad', 'h22-uv', 'h23-solar_rad', 'h23-uv', 'evento', 'fin', 'inicio',
                'cnt_usus', 'DEP', 'MUN', 'FECHA', 'NIVEL_C', 'VALOR_C', 'TRAMOS_AGUAS_ABAJO', 'EQUIPOS_PUNTOS',
                'PUNTOS_POLIGONO', 'LONGITUD2', 'LATITUD2', 'FECHA_C','TRAMOS_AGUAS_ABAJO_CODES','ORDER_'],
            inplace=True, axis=1)

    # Definir la variable objetivo y eliminarla del conjunto de características
    target = ['SAIFI', 'SAIDI', 'duracion_h']
    y1 = Xdata[target].values
    Xdata.drop(target, axis=1, inplace=True)
    y1=y1[:,0:1]
    df = Xdata.copy()

    # Identificar columnas numéricas y categóricas
    NUMERIC_COLUMNS = df.select_dtypes(include=['number']).columns.tolist()
    CATEGORICAL_COLUMNS = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Crear diccionarios para guardar LabelEncoders y valores máximos
    label_encoders = {}
    max_values = {}
    categorical_dims = {}
    # Rellenar valores faltantes y guardar los valores máximos
    for col in NUMERIC_COLUMNS:
        max_value = df[col].max()
        max_values[col] = max_value
        df[col].fillna(-10 * max_value, inplace=True)

    # Codificar variables categóricas y guardar LabelEncoders
    for col in CATEGORICAL_COLUMNS:
        l_enc = LabelEncoder()
        l_enc.fit(df[col].astype(str).fillna("no aplica"))
        df[col] = l_enc.transform(df[col].astype(str).fillna("no aplica"))
        label_encoders[col] = l_enc
        categorical_dims[col] = len(l_enc.classes_)
    # Crear lista de características
    unused_feat = []
    features = [col for col in df.columns if col not in unused_feat + target]
    # Preparar datos
    X = df[features].values.astype('float32')
    y = y1.astype('float32')
    # Crear categorías basadas en percentiles
    percentiles = np.percentile(y[:, 0], [33.33, 66.66])
    y_categorized = np.digitize(y[:, 0:1].flatten(), bins=percentiles).astype(int)
    scolumns = list(
        set(redmt.columns)
        .union(set(apoyos.columns))
        .union(set(trafos.columns))
        .union(set(switches.columns))
    )

    # Crear categorías basadas en percentiles
    percentiles = np.percentile(y[:, 0], [33.33, 66.66])
    y_categorized = df1['NIVEL_C'].values#np.digitize(y[:, 0:1].flatten(), bins=percentiles).astype(int)

    # Escalar la variable objetivo
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y)

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_scaled, test_size=0.2, random_state=42, stratify=y_categorized)

    # Dividir entrenamiento en entrenamiento y validación
    percentiles_t = np.percentile(y_train[:, 0], [25, 50,75])
    y_categorized_t = np.digitize(y_train[:, 0:1].flatten(), bins=percentiles_t).astype(int)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_categorized_t)


    # Crear la lista de opciones
    options = [{"label": " ", "value": ""}] + [{"label": col, "value": col} for col in df1.columns.to_list()]

    if os.path.exists("./options/criterias_2.json"):
        os.remove("./options/criterias_2.json")

    # Guardar en un archivo JSON
    with open("./options/criterias_2.json", "w") as f:
        json.dump(options, f)

    loaded_clf = torch.load(os.path.join("..", "data", "model.pth"))

    return trafos, apoyos, switches, redmt, super_eventos, descargas, vegetacion, df, df1, label_encoders, scolumns, NUMERIC_COLUMNS, max_values, loaded_clf

def map_folium(trafos_seleccionado, apoyos_seleccionado, switches_seleccionado, redmt_seleccionado, super_eventos_seleccionado, descargas_seleccionado, vegetacion_seleccionado, condicion):
    

    if condicion == '':

        # Crear un mapa centrado en la media de las coordenadas de los circuitos
        map_center = [apoyos_seleccionado.LATITUD.mean(), apoyos_seleccionado.LONGITUD.mean()]
        mapa = folium.Map(location=map_center, zoom_start=15, width='100%', height='100%')

        for idx, row in super_eventos_seleccionado.iterrows():

            match row.tipo_equi_ope:

                case 'interruptor':
                    location = [row.LATITUD, row.LONGITUD]
                    # Agregar un marcador con ícono de advertencia en rojo
                    folium.Marker(
                        location=location,
                        popup=f"Evento \n Equipo opero: {row.equipo_ope} \n Tipo equipo: {row.tipo_equi_ope} \n Circuito opero: {row.cto_equi_ope} \n Tipo elemento: {row.tipo_elemento} \n Duracion: {row.duracion_h:.4f} \n Causa: {row.causa} \n Cantidad usuarios: {row.cnt_usus} \n SAIDI: {row.SAIDI:.4f} \n Inicio: {row.inicio}\n Fin: {row.fin}",
                        icon=folium.Icon(icon="exclamation-triangle", prefix="fa", color="red")  # Ícono de advertencia rojo
                    ).add_to(mapa)

                case 'tramo de linea':

                    location = [row.LATITUD, row.LONGITUD]
                    # Agregar un marcador con ícono de advertencia en rojo
                    folium.Marker(
                        location=location,
                        popup=f"Evento \n Equipo opero: {row.equipo_ope} \n Tipo equipo: {row.tipo_equi_ope} \n Circuito opero: {row.cto_equi_ope} \n Tipo elemento: {row.tipo_elemento} \n Duracion: {row.duracion_h:.4f} \n Causa: {row.causa} \n Cantidad usuarios: {row.cnt_usus} \n SAIDI: {row.SAIDI:.4f} \n Inicio: {row.inicio} \n Fin: {row.fin}",
                        icon=folium.Icon(icon="exclamation-triangle", prefix="fa", color="red")  # Ícono de advertencia rojo
                    ).add_to(mapa)

                case 'transformador':

                    location = [row.LATITUD, row.LONGITUD]
                    # Agregar un marcador con ícono de advertencia en rojo
                    folium.Marker(
                        location=location,
                        popup=f"Evento \n Equipo opero: {row.equipo_ope} \n Tipo equipo: {row.tipo_equi_ope} \n Circuito opero: {row.cto_equi_ope} \n Tipo elemento: {row.tipo_elemento} \n Duracion: {row.duracion_h:.4f} \n Causa: {row.causa} \n Cantidad usuarios: {row.cnt_usus} \n SAIDI: {row.SAIDI:.4f} \n Inicio: {row.inicio} \n Fin: {row.fin}",
                        icon=folium.Icon(icon="exclamation-triangle", prefix="fa", color="red")  # Ícono de advertencia rojo
                    ).add_to(mapa)


        # Crear la leyenda HTML
        legend_html = """
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 90px; height: 115px; 
                    background-color: white; border:2px solid black; 
                    z-index:9999; font-size:12px; padding: 8px; opacity: 0.7;">
            <i style="background-color:blue; width: 15px; height: 15px; display: inline-block;"></i> Apoyos<br>
            <i style="background-color:green; width: 15px; height: 15px; display: inline-block;"></i> Trafos<br>
            <i style="background-color:brown; width: 15px; height: 15px; display: inline-block;"></i> Switches<br>
            <i style="background-color:black; width: 15px; height: 15px; display: inline-block;"></i> Red MT<br>
            <i style="background-color:red; width: 15px; height: 15px; display: inline-block;"></i> Eventos<br>
        </div>
        """

        # Añadir la leyenda directamente al mapa
        mapa.get_root().html.add_child(folium.Element(legend_html))

        for idx, row in redmt_seleccionado.iterrows(): 
            # Dibujar los segmentos en el orden deseado
            linea = folium.PolyLine(
                locations=[(row.LATITUD,row.LONGITUD), (row.LATITUD2,row.LONGITUD2)],
                color="black",
                weight=1.5,
                opacity=1
            )
            popup = folium.Popup(f"Tramo de linea \n Material conductor: {row.MATERIALCONDUCTOR} \n Tipo conductor: {row.TIPOCONDUCTOR} \n Largo: {row.LENGTH} \n Calibre conductor: {row.CALIBRECONDUCTOR} \n Guarda conductor:{row.GUARDACONDUCTOR} \n Neutro conductor:{row.NEUTROCONDUCTOR} \n Calibre neutro:{row.CALIBRENEUTRO} \n Capacidad: {row.CAPACITY} \n Resistencia: {row.RESISTANCE:.4f} \n Acometida conductor: {row.ACOMETIDACONDUCTOR}")
            linea.add_child(popup)

            # Añadir la polyline al mapa
            linea.add_to(mapa)

        # Agregar los apoyos al mapa
        for idx, row in apoyos_seleccionado.iterrows():
            lat = row.LATITUD # Coordenadas en y
            lon = row.LONGITUD # Coordenadas en x
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color='blue',
                fill=True,
                fill_color='cyan',
                fill_opacity=0.6,
                popup=f"Apoyo Propietario: {row.TOWNER} \n Tipo: {row.TIPO} \n Clase: {row.CLASE} \n Material: {row.MATERIAL} \n Longitud: {row.LONG_APOYO} \n Tierra pie: {row.TIERRA_PIE} \n Vientos: {row.VIENTOS}"
            ).add_to(mapa)

        # Agregar los trafos al mapa
        for idx, row in trafos_seleccionado.iterrows():
            lat = row.LATITUD # Coordenadas en y
            lon = row.LONGITUD # Coordenadas en x
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.6,
                popup=f"Trafo Fase: {row.PHASES} \n Propietario: {row.OWNER1} \n Impedancia: {row.IMPEDANCE} \n Marca: {row.MARCA} \n Fecha fabricacion: {row.DATE_FAB[:10]} \n Tipo subestación: {row.TIPO_SUB} \n KVA: {row.KVA} \n KV1: {row.KV1} \n FPARENT:{row.FPARENT}"
            ).add_to(mapa)

        # Agregar los switches al mapa
        for idx, row in switches_seleccionado.iterrows():
            lat = row.LATITUD # Coordenadas en y
            lon = row.LONGITUD # Coordenadas en x
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color='brown',
                fill=True,
                fill_color='brown',
                fill_opacity=0.6,
                popup=f"Switche Fase: {row.PHASES} \n Codigo assembly: {row.ASSEMBLY} \n KV: {row.KV} \n Estado: {row.STATE}"
            ).add_to(mapa)
        
        mapa_html = mapa._repr_html_()

        return mapa_html

    elif condicion == 'DESCARGAS':

        # Crear un mapa centrado en la media de las coordenadas de los circuitos
        map_center = [switches_seleccionado.LATITUD.mean(), switches_seleccionado.LONGITUD.mean()]
        mapa = folium.Map(location=map_center, zoom_start=11, tiles='None', attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors') 

        # Agregar el estilo satelital de Esri
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Esri Satellite",
            overlay=False,
            control=False  # Desactiva el control de capas para que solo muestre Esri
        ).add_to(mapa)

        for idx, row in descargas_seleccionado.iterrows():
            location = [row.LATITUD, row.LONGITUD]
            # Agregar un marcador con ícono de advertencia en rojo
            folium.CircleMarker(
                location=location,
                radius=float(row.ERROR) * (111 + 111 * math.cos(math.radians(row.LATITUD))) / 2,
                popup=f"Rayo ID: {row.ID} Altitud: {row.ALTITUD} Tipo: {row.TIPO} Corriente: {row.CORRIENTE} Distancia a nodo: {row.DISTANCIA_A_NODO} Fecha: {row.FECHA}",
                color='cyan',
                fill=True,
                fill_color='cyan',
                fill_opacity=0.3,
                icon=folium.Icon(icon="exclamation-triangle", prefix="fa", color="red")  # Ícono de advertencia rojo
            ).add_to(mapa)

       # Crear la leyenda HTML
        legend_html = """
        <div style="position: fixed; 
                    top: 15px; right: 10px; width: 90px; height: 115px; 
                    background-color: white; border:2px solid black; 
                    z-index:9999; font-size:12px; padding: 8px; opacity: 0.7;">
            <i style="background-color:blue; width: 15px; height: 15px; display: inline-block;"></i> Apoyos<br>
            <i style="background-color:green; width: 15px; height: 15px; display: inline-block;"></i> Trafos<br>
            <i style="background-color:brown; width: 15px; height: 15px; display: inline-block;"></i> Switches<br>
            <i style="background-color:black; width: 15px; height: 15px; display: inline-block;"></i> Red MT<br>
            <i style="background-color:cyan; width: 15px; height: 15px; display: inline-block;"></i> Rayos<br>
        </div>
        """

        # Añadir la leyenda directamente al mapa
        mapa.get_root().html.add_child(folium.Element(legend_html))

        for idx, row in redmt_seleccionado.iterrows(): 
            # Dibujar los segmentos en el orden deseado
            linea = folium.PolyLine(
                locations=[(row.LATITUD,row.LONGITUD), (row.LATITUD2,row.LONGITUD2)],
                color="black",
                weight=1.5,
                opacity=1
            )
            popup = folium.Popup(f"Tramo de linea \n Material conductor: {row.MATERIALCONDUCTOR} \n Tipo conductor: {row.TIPOCONDUCTOR} \n Largo: {row.LENGTH} \n Calibre conductor: {row.CALIBRECONDUCTOR} \n Guarda conductor:{row.GUARDACONDUCTOR} \n Neutro conductor:{row.NEUTROCONDUCTOR} \n Calibre neutro:{row.CALIBRENEUTRO} \n Capacidad: {row.CAPACITY} \n Resistencia: {row.RESISTANCE:.4f} \n Acometida conductor: {row.ACOMETIDACONDUCTOR}")
            linea.add_child(popup)

            # Añadir la polyline al mapa
            linea.add_to(mapa)

       # Agregar los apoyos al mapa
        for idx, row in apoyos_seleccionado.iterrows():
            lat = row.LATITUD # Coordenadas en y
            lon = row.LONGITUD # Coordenadas en x
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color='blue',
                fill=True,
                fill_color='cyan',
                fill_opacity=0.6,
                popup=f"Apoyo Propietario: {row.TOWNER} \n Tipo: {row.TIPO} \n Clase: {row.CLASE} \n Material: {row.MATERIAL} \n Longitud: {row.LONG_APOYO} \n Tierra pie: {row.TIERRA_PIE} \n Vientos: {row.VIENTOS}"
            ).add_to(mapa)

        # Agregar los trafos al mapa
        for idx, row in trafos_seleccionado.iterrows():
            lat = row.LATITUD # Coordenadas en y
            lon = row.LONGITUD # Coordenadas en x
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.6,
                popup=f"Trafo Fase: {row.PHASES} \n Propietario: {row.OWNER1} \n Impedancia: {row.IMPEDANCE} \n Marca: {row.MARCA} \n Fecha fabricacion: {row.DATE_FAB[:10]} \n Tipo subestación: {row.TIPO_SUB} \n KVA: {row.KVA} \n KV1: {row.KV1}"
            ).add_to(mapa)

        # Agregar los switches al mapa
        for idx, row in switches_seleccionado.iterrows():
            lat = row.LATITUD # Coordenadas en y
            lon = row.LONGITUD # Coordenadas en x
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color='brown',
                fill=True,
                fill_color='brown',
                fill_opacity=0.6,
                popup=f"Switche Fase: {row.PHASES} \n Codigo assembly: {row.ASSEMBLY} \n KV: {row.KV} \n Estado: {row.STATE}"
            ).add_to(mapa)

        mapa_html = mapa._repr_html_()

        return mapa_html
    
    elif condicion == 'VEGETACION':

        # Crear un mapa centrado en la media de las coordenadas de los circuitos
        map_center = [
            switches_seleccionado.LATITUD.mean(),
            switches_seleccionado.LONGITUD.mean()
        ]
        mapa = folium.Map(
            location=map_center, 
            zoom_start=11, 
            tiles='None', 
            attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors'
        )

        # Agregar el estilo satelital de Esri
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Esri Satellite",
            overlay=False,
            control=False  # Desactiva el control de capas para que solo muestre Esri
        ).add_to(mapa)

       # Crear la leyenda HTML
        legend_html = """
        <div style="position: fixed; 
                    top: 15px; right: 10px; width: 105px; height: 120px; 
                    background-color: white; border:2px solid black; 
                    z-index:9999; font-size:12px; padding: 8px; opacity: 0.7;">
            <i style="background-color:blue; width: 15px; height: 15px; display: inline-block;"></i> Apoyos<br>
            <i style="background-color:green; width: 15px; height: 15px; display: inline-block;"></i> Trafos<br>
            <i style="background-color:brown; width: 15px; height: 15px; display: inline-block;"></i> Switches<br>
            <i style="background-color:black; width: 15px; height: 15px; display: inline-block;"></i> Red MT<br>
            <i style="background-color:red; width: 15px; height: 15px; display: inline-block;"></i> Vegetacion<br>
        </div>
        """

        # Añadir la leyenda directamente al mapa
        mapa.get_root().html.add_child(folium.Element(legend_html))

        for idx, row in redmt_seleccionado.iterrows(): 
            # Dibujar los segmentos en el orden deseado
            linea = folium.PolyLine(
                locations=[(row.LATITUD,row.LONGITUD), (row.LATITUD2,row.LONGITUD2)],
                color="black",
                weight=1.5,
                opacity=1
            )
            popup = folium.Popup(f"Tramo de linea \n Material conductor: {row.MATERIALCONDUCTOR} \n Tipo conductor: {row.TIPOCONDUCTOR} \n Largo: {row.LENGTH} \n Calibre conductor: {row.CALIBRECONDUCTOR} \n Guarda conductor:{row.GUARDACONDUCTOR} \n Neutro conductor:{row.NEUTROCONDUCTOR} \n Calibre neutro:{row.CALIBRENEUTRO} \n Capacidad: {row.CAPACITY} \n Resistencia: {row.RESISTANCE:.4f} \n Acometida conductor: {row.ACOMETIDACONDUCTOR}")
            linea.add_child(popup)

            # Añadir la polyline al mapa
            linea.add_to(mapa)

        # Agregar los apoyos al mapa
        for idx, row in apoyos_seleccionado.iterrows():
            lat = row.LATITUD # Coordenadas en y
            lon = row.LONGITUD # Coordenadas en x
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color='blue',
                fill=True,
                fill_color='cyan',
                fill_opacity=0.6,
                popup=f"Apoyo Propietario: {row.TOWNER} \n Tipo: {row.TIPO} \n Clase: {row.CLASE} \n Material: {row.MATERIAL} \n Longitud: {row.LONG_APOYO} \n Tierra pie: {row.TIERRA_PIE} \n Vientos: {row.VIENTOS}"
            ).add_to(mapa)

        # Agregar los trafos al mapa
        for idx, row in trafos_seleccionado.iterrows():
            lat = row.LATITUD # Coordenadas en y
            lon = row.LONGITUD # Coordenadas en x
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.6,
                popup=f"Trafo Fase: {row.PHASES} \n Propietario: {row.OWNER1} \n Impedancia: {row.IMPEDANCE} \n Marca: {row.MARCA} \n Fecha fabricacion: {row.DATE_FAB[:10]} \n Tipo subestación: {row.TIPO_SUB} \n KVA: {row.KVA} \n KV1: {row.KV1}"
            ).add_to(mapa)

        # Agregar los switches al mapa
        for idx, row in switches_seleccionado.iterrows():
            lat = row.LATITUD # Coordenadas en y
            lon = row.LONGITUD # Coordenadas en x
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color='brown',
                fill=True,
                fill_color='brown',
                fill_opacity=0.6,
                popup=f"Switche Fase: {row.PHASES} \n Codigo assembly: {row.ASSEMBLY} \n KV: {row.KV} \n Estado: {row.STATE}"
            ).add_to(mapa)
        
        riesgo_valores = {'Alto': 3, 'Medio': 2, 'Bajo': 1}

        # Diccionario de rangos normalizados para cada nivel de riesgo
        riesgo_valores_normalizados = {'Bajo': (0.0, 0.33), 'Medio': (0.33, 0.66), 'Alto': (0.66, 1.0)}

        # Gradiente general (amarillo, naranja, rojo)
        gradient_general = {
            0.0: 'yellow',  # Riesgo bajo
            0.33: 'orange', # Riesgo medio
            0.66: '#FF4500', # Naranja oscuro (inicio del riesgo alto)
            1.0: 'red'      # Riesgo máximo
        }

        # Crear una lista de datos con intensidades normalizadas
        heat_data = []
        for _, row in vegetacion_seleccionado.iterrows():
            nivel_riesgo = row['NIVEL_RIES']
            rango_min, rango_max = riesgo_valores_normalizados[nivel_riesgo]
            intensidad_normalizada = rango_min + (riesgo_valores[nivel_riesgo] - 1) * (rango_max - rango_min) / 2
            heat_data.append([row['LATITUD'], row['LONGITUD'], intensidad_normalizada])

        # Agregar el HeatMap al mapa
        HeatMap(
            data=heat_data,
            min_opacity=0.5,
            radius=15,
            gradient=gradient_general
        ).add_to(mapa)

        mapa_html = mapa._repr_html_()

        return mapa_html

        '''
        # Diccionario para colores de los CircleMarker según nivel de riesgo
        riesgo_colores = {'Alto': 'red', 'Medio': 'orange', 'Bajo': 'yellow'}

        # Agregar CircleMarker para cada punto
        for _, row in vegetacion_seleccionado.iterrows():
            folium.CircleMarker(
                location=[row['LATITUD'], row['LONGITUD']],
                radius=10,  # Radio del marcador
                color=riesgo_colores[row['NIVEL_RIES']],  # Color basado en el nivel de riesgo
                fill=True,
                fill_color=riesgo_colores[row['NIVEL_RIES']],
                fill_opacity=0.6,
                popup=f"Nivel de riesgo: {row['NIVEL_RIES']}"  # Popup con información
            ).add_to(mapa)'''
        
    elif condicion == 'CRITICIDAD':

        # Crear un mapa centrado en la media de las coordenadas de los circuitos
        map_center = [
            switches_seleccionado.LATITUD.mean(),
            switches_seleccionado.LONGITUD.mean()
        ]
        mapa = folium.Map(
            location=map_center, 
            zoom_start=11, 
            tiles='None', 
            attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors'
        )

        # Agregar el estilo satelital de Esri
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Esri Satellite",
            overlay=False,
            control=False  # Desactiva el control de capas para que solo muestre Esri
        ).add_to(mapa)

       # Crear la leyenda HTML
        legend_html = """
        <div style="position: fixed; 
                    top: 15px; right: 10px; width: 135px; height: 170px; 
                background-color: white; border:2px solid black; 
                z-index:9999; font-size:12px; padding: 8px; opacity: 0.7;">
        <i style="background-color:blue; width: 15px; height: 15px; display: inline-block;"></i> Apoyos<br>
        <i style="background-color:green; width: 15px; height: 15px; display: inline-block;"></i> Trafos<br>
        <i style="background-color:purple; width: 15px; height: 15px; display: inline-block;"></i> Switches<br>
        <i style="background-color:black; width: 15px; height: 15px; display: inline-block;"></i> Red MT<br>
        <i style="background-color:green; width: 15px; height: 15px; display: inline-block;"></i> CR baja<br>
        <i style="background-color:yellow; width: 15px; height: 15px; display: inline-block;"></i> CR media<br>
        <i style="background-color:orange; width: 15px; height: 15px; display: inline-block;"></i> CR alta<br>
        <i style="background-color:red; width: 15px; height: 15px; display: inline-block;"></i> CR muy alta<br>
        </div>
        """

        # Añadir la leyenda directamente al mapa
        mapa.get_root().html.add_child(folium.Element(legend_html))

        nodos_aguas_abajo = set()
        puntos_equipos = list()

        for evento in super_eventos_seleccionado['evento'].values:
            df = super_eventos_seleccionado[super_eventos_seleccionado['evento'] == evento]
            polygon_puntos = eval(df['PUNTOS_POLIGONO'].values[0])
            # Dibuja el polígono en el mapa
            match df['NIVEL_C'].values[0]:
                case 0:
                    color = 'green'
                case 1:
                    color = 'yellow'
                case 2:
                    color = 'orange'
                case _:
                    color = 'red'
            if len(polygon_puntos) != 1:
                folium.Polygon(locations=polygon_puntos, color=color, fill=True, fill_opacity=0.3, weight=0).add_to(mapa)

            puntos_tramos = eval(df['TRAMOS_AGUAS_ABAJO'].values[0])
            nodos_aguas_abajo.update(puntos_tramos)
            
            puntos_verdes = eval(df['EQUIPOS_PUNTOS'].values[0]) 
            puntos_equipos.extend(puntos_verdes)


        for _, row in redmt_seleccionado.iterrows():
            point1 = (row["LATITUD"], row["LONGITUD"])
            point2 = (row["LATITUD2"], row["LONGITUD2"])
            
            if (point1 in nodos_aguas_abajo and point2 in nodos_aguas_abajo):
                color = 'black'
            else:
                color = 'gray'
               
            folium.PolyLine(
                [point1, point2],
                color=color,
                weight=3,
                opacity=1.0,
                popup=f"Tramo de linea \n Material conductor: {row.MATERIALCONDUCTOR} \n Tipo conductor: {row.TIPOCONDUCTOR} \n Largo: {row.LENGTH} \n Calibre conductor: {row.CALIBRECONDUCTOR} \n Guarda conductor:{row.GUARDACONDUCTOR} \n Neutro conductor:{row.NEUTROCONDUCTOR} \n Calibre neutro:{row.CALIBRENEUTRO} \n Capacidad: {row.CAPACITY} \n Resistencia: {row.RESISTANCE:.4f} \n Acometida conductor: {row.ACOMETIDACONDUCTOR}"
            ).add_to(mapa)

        
        for _, row in trafos_seleccionado.iterrows():
            point1 = (row["LATITUD"], row["LONGITUD"])
            if (point1 in puntos_equipos):
                color = 'green'
            else:
                color = 'gray'
            folium.CircleMarker(
                location=(row["LATITUD"], row["LONGITUD"]),
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=1.0,
                popup=f"Trafo Fase: {row.PHASES} \n Propietario: {row.OWNER1} \n Impedancia: {row.IMPEDANCE} \n Marca: {row.MARCA} \n Fecha fabricacion: {row.DATE_FAB[:10]} \n Tipo subestación: {row.TIPO_SUB} \n KVA: {row.KVA} \n KV1: {row.KV1}"
            ).add_to(mapa)

        for _, row in apoyos_seleccionado.iterrows():
            point1 = (row["LATITUD"], row["LONGITUD"])
            if (point1 in puntos_equipos):
                color = 'blue'
            else:
                color = 'gray'
            folium.CircleMarker(
                location=(row["LATITUD"], row["LONGITUD"]),
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=1.0,
                popup=f"Apoyo Propietario: {row.TOWNER} \n Tipo: {row.TIPO} \n Clase: {row.CLASE} \n Material: {row.MATERIAL} \n Longitud: {row.LONG_APOYO} \n Tierra pie: {row.TIERRA_PIE} \n Vientos: {row.VIENTOS}"
            ).add_to(mapa)

        for _, row in switches_seleccionado.iterrows():
            point1 = (row["LATITUD"], row["LONGITUD"])
            if (point1 in puntos_equipos):
                color = 'purple'
            else:
                color = 'gray'
            folium.CircleMarker(
                location=(row["LATITUD"], row["LONGITUD"]),
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=1.0,
                popup=f"Switche Fase: {row.PHASES} \n Codigo assembly: {row.ASSEMBLY} \n KV: {row.KV} \n Estado: {row.STATE}"
            ).add_to(mapa)
       

        mapa_html = mapa._repr_html_()
    
        return mapa_html


def map_folium_2(trafos_seleccionado, apoyos_seleccionado, switches_seleccionado, redmt_seleccionado, super_eventos_seleccionado, ind, df, df1, label_encoders, scolumns, NUMERIC_COLUMNS, max_values, loaded_clf):

    a1,a1_df = process_dataframe(trafos_seleccionado, df, label_encoders, df1, ind=ind,tip=2,s=1,scolumns=scolumns, NUMERIC_COLUMNS=NUMERIC_COLUMNS, max_values=max_values)
    a2,a2_df = process_dataframe(switches_seleccionado, df, label_encoders, df1, ind=ind,tip=0,s=1,scolumns=scolumns, NUMERIC_COLUMNS=NUMERIC_COLUMNS, max_values=max_values)
    a3,a3_df = process_dataframe(redmt_seleccionado, df, label_encoders, df1, ind=ind,tip=1,s=0,scolumns=scolumns, NUMERIC_COLUMNS=NUMERIC_COLUMNS, max_values=max_values)
    a4,a4_df = process_dataframe(apoyos_seleccionado, df, label_encoders, df1, ind=ind,tip=2,s=1,scolumns=scolumns, NUMERIC_COLUMNS=NUMERIC_COLUMNS, max_values=max_values)
    columns=df.columns
    arrays_to_concatenate = [arr for arr in (a1, a2, a3, a4) if arr.size > 0]
    a = np.concatenate(arrays_to_concatenate, axis=0)
    y_e=loaded_clf.predict(a)
    y_e=y_e.flatten()
    top_3_indices = np.argsort(y_e)[-3:][::-1]
    _,masks=loaded_clf.explain(a[top_3_indices])
    mask3=np.array(list(masks.values())).sum(axis=0)
    # Definir los nombres de los equipos
    equipo_nombres = {0: 'interruptor', 1: 'tramo de linea', 2: 'transformador', 'apoyo': 'apoyo'}

    # Calcular los límites de los índices para cada conjunto
    a1_end = len(a1)
    a2_end = a1_end + len(a2)
    a3_end = a2_end + len(a3)
    a4_start = a3_end  # El inicio de `a4` es el final de `a3`

    # Generar el diccionario final para las 3 muestras más relevantes
    resultados = {}
    columnas = df.columns.tolist()  # Obtener lista de nombres de columnas

    # Encontrar las posiciones de LATITUD y LONGITUD
    pos_latitud = columnas.index('LATITUD') if 'LATITUD' in columnas else None
    pos_longitud = columnas.index('LONGITUD') if 'LONGITUD' in columnas else None
    a_df=pd.concat((a1_df,a2_df,a3_df,a4_df),ignore_index=True)
    aux_eq=[]
    # Iterar sobre las 3 muestras más relevantes
    for idx, muestra_idx in enumerate(top_3_indices):
        # Determinar el tipo de equipo según el índice
        if muestra_idx >= a4_start:
            tipo_equipo = 'apoyo'  # Índices correspondientes a `a4`
        else:
            tipo_equipo = equipo_nombres.get(a[muestra_idx, 1], 'desconocido')  # Columna 1 determina el tipo
        aux_eq.append(tipo_equipo)
        # Obtener las 10 columnas más relevantes según `mask3`
        top_10_indices = np.argsort(mask3[idx])[-20:][::-1]  # Ordenar en descendente
        top_10_columnas = [columns[i] for i in top_10_indices]
        # Obtener los valores correspondientes a las 10 columnas más relevantes
        top_10_valores = [a_df.iloc[idx, i] for i in top_10_indices]

        # Crear un diccionario temporal de las variables y sus valores
        temp_variables = dict(zip(top_10_columnas, top_10_valores))

            # Filtrar variables que pertenecen al tipo de equipo actual
        if tipo_equipo=='apoyo':
            allowed_columns = set(df.columns).difference(set(trafos_seleccionado.columns)).difference(set(switches_seleccionado.columns)).difference(set(redmt_seleccionado.columns))
        elif tipo_equipo=='interruptor':
            allowed_columns = set(df.columns).difference(set(trafos_seleccionado.columns)).difference(set(redmt_seleccionado.columns)).difference(set(apoyos_seleccionado.columns))
        elif tipo_equipo=='tramo de linea':
            allowed_columns = set(df.columns).difference(set(trafos_seleccionado.columns)).difference(set(switches_seleccionado.columns)).difference(set(apoyos_seleccionado.columns))
        elif tipo_equipo=='transformador':
            allowed_columns = set(df.columns).difference(set(switches_seleccionado.columns)).difference(set(redmt_seleccionado.columns)).difference(set(apoyos_seleccionado.columns))
        be=filtered_variables = {k: v for k, v in temp_variables.items() if k in allowed_columns}

        # Eliminar solo las claves cuyos valores sean NaN, None o cadenas vacías
        filtered_variables = {
        k: v for k, v in filtered_variables.items()
        if not math.isnan(v)
        }

        #filtered_variables = {k: v for k, v in filtered_variables.items() if isinstance(v, (int, float)) and not math.isnan(v)}
        resultados[f'muestra_{idx+1}'] = {
            'tipo_equipo': tipo_equipo,
            'top_5_variables': dict(list(filtered_variables.items())[:10]),  # Top 10 relevantes después del filtro
            'posición': (a[muestra_idx, pos_latitud], a[muestra_idx, pos_longitud])
        }

    resultado = []
    for _, info in resultados.items():
        resultado.append([switches_seleccionado.loc[(switches_seleccionado['LATITUD'] == info['posición'][0]) & (switches_seleccionado['LONGITUD'] == info['posición'][1]), 'equipo_ope'].unique(),'switch'])
        resultado.append([apoyos_seleccionado.loc[(apoyos_seleccionado['LATITUD'] == info['posición'][0]) & (apoyos_seleccionado['LONGITUD'] == info['posición'][1]), 'equipo_ope'].unique(),'apoyo'])
        resultado.append([trafos_seleccionado.loc[(trafos_seleccionado['LATITUD'] == info['posición'][0]) & (trafos_seleccionado['LONGITUD'] == info['posición'][1]), 'equipo_ope'].unique(),'transformador'])
        resultado.append([redmt_seleccionado.loc[(redmt_seleccionado['LATITUD'] == info['posición'][0]) & (redmt_seleccionado['LONGITUD'] == info['posición'][1]), 'equipo_ope'].unique(),'tramo_red'])
    
    equipos_criticos = [item for item in resultado if item[0].size > 0]
    codigo_equipos_criticos = [item[0][0] for item in equipos_criticos]
    tipo_equipos_criticos = [item[1] for item in equipos_criticos] 

    # Construcción del diccionario con strings
    resultado_dict = {}

    for (muestra, info), codigo in zip(resultados.items(), codigo_equipos_criticos):
        resultado_dict[codigo] = {
            "Tipo_de_equipo": info["tipo_equipo"],
            "top_5": {clave: str(valor) for clave, valor in info["top_5_variables"].items()}  # Convertir valores a string
        }


    # Procesar el JSON
    resultado_json = procesar_json(resultado_dict)

    # Guardar el resultado en un archivo JSON
    with open("equipos_filtrados", "w", encoding="utf-8") as archivo:
        json.dump(resultado_json, archivo, ensure_ascii=False, indent=4)


    # Crear un mapa centrado en la media de las coordenadas de los circuitos
    map_center = [
        switches_seleccionado.LATITUD.mean(),
        switches_seleccionado.LONGITUD.mean()
    ]
    mapa = folium.Map(
        location=map_center, 
        zoom_start=11, 
        tiles='None', 
        attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors'
    )

    # Agregar el estilo satelital de Esri
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        overlay=False,
        control=False  # Desactiva el control de capas para que solo muestre Esri
    ).add_to(mapa)

    # Crear la leyenda HTML
    legend_html = """
    <div style="position: fixed; 
                top: 15px; right: 10px; width: 135px; height: 170px; 
                background-color: white; border:2px solid black; 
                z-index:9999; font-size:12px; padding: 8px; opacity: 0.7;">
        <i style="background-color:blue; width: 15px; height: 15px; display: inline-block;"></i> Apoyos<br>
        <i style="background-color:green; width: 15px; height: 15px; display: inline-block;"></i> Trafos<br>
        <i style="background-color:purple; width: 15px; height: 15px; display: inline-block;"></i> Switches<br>
        <i style="background-color:black; width: 15px; height: 15px; display: inline-block;"></i> Red MT<br>
        <i style="background-color:green; width: 15px; height: 15px; display: inline-block;"></i> CR baja<br>
        <i style="background-color:yellow; width: 15px; height: 15px; display: inline-block;"></i> CR media<br>
        <i style="background-color:orange; width: 15px; height: 15px; display: inline-block;"></i> CR alta<br>
        <i style="background-color:red; width: 15px; height: 15px; display: inline-block;"></i> CR muy alta<br>
    </div>
    """

    # Añadir la leyenda directamente al mapa
    mapa.get_root().html.add_child(folium.Element(legend_html))

    nodos_aguas_abajo = set()
    puntos_equipos = list()

    for evento in super_eventos_seleccionado['evento'].values:
        df_2 = super_eventos_seleccionado[super_eventos_seleccionado['evento'] == evento]
        polygon_puntos = eval(df_2['PUNTOS_POLIGONO'].values[0])
        # Dibuja el polígono en el mapa
        match df_2['NIVEL_C'].values[0]:
            case 0:
                color = 'green'
            case 1:
                color = 'yellow'
            case 2:
                color = 'orange'
            case 3:
                color = 'red'
        if len(polygon_puntos) != 1:
            folium.Polygon(locations=polygon_puntos, color=color, fill=True, fill_opacity=0.3, weight=0).add_to(mapa)

        puntos_tramos = eval(df_2['TRAMOS_AGUAS_ABAJO'].values[0])
        nodos_aguas_abajo.update(puntos_tramos)
        
        puntos_verdes = eval(df_2['EQUIPOS_PUNTOS'].values[0]) 
        puntos_equipos.extend(puntos_verdes)


    for _, row in redmt_seleccionado.iterrows():
        point1 = (row["LATITUD"], row["LONGITUD"])
        point2 = (row["LATITUD2"], row["LONGITUD2"])
        
        if (point1 in nodos_aguas_abajo and point2 in nodos_aguas_abajo):
            color = 'black'
        else:
            color = 'gray'
            
        folium.PolyLine(
            [point1, point2],
            color=color,
            weight=3,
            opacity=1.0,
            popup=f"Tramo de linea \n Material conductor: {row.MATERIALCONDUCTOR} \n Tipo conductor: {row.TIPOCONDUCTOR} \n Largo: {row.LENGTH} \n Calibre conductor: {row.CALIBRECONDUCTOR} \n Guarda conductor:{row.GUARDACONDUCTOR} \n Neutro conductor:{row.NEUTROCONDUCTOR} \n Calibre neutro:{row.CALIBRENEUTRO} \n Capacidad: {row.CAPACITY} \n Resistencia: {row.RESISTANCE:.4f} \n Acometida conductor: {row.ACOMETIDACONDUCTOR}"
        ).add_to(mapa)

    # Crear el dataframe `aporte_SAIDI`
    aporte_SAIDI = a_df[['LATITUD', 'LONGITUD']].copy()
    aporte_SAIDI['SAIDI'] = y_e

    # Reemplazar valores <= 0 con un pequeño número positivo para evitar problemas en el logaritmo
    aporte_SAIDI.loc[aporte_SAIDI['SAIDI'] == 0, 'SAIDI'] = np.nan  # Convierte 0 a NaN
    aporte_SAIDI['SAIDI'] = aporte_SAIDI['SAIDI'].fillna(1e-6)       # Sustituye NaN por un valor pequeño

    # Aplicar la transformación logarítmica (logaritmo base 10)
    aporte_SAIDI['log_transformada'] = np.log10(aporte_SAIDI['SAIDI'])

    # 2. Escalar los datos al rango [2, 5] usando MinMaxScaler
    scaler = MinMaxScaler(feature_range=(5, 10))
    aporte_SAIDI['escalada'] = scaler.fit_transform(aporte_SAIDI[['log_transformada']])
    
    for _, row in trafos_seleccionado.iterrows():
        point1 = (row["LATITUD"], row["LONGITUD"])
        radio = 5
        if (point1 in puntos_equipos):
            color = 'green'
        else:
            color = 'gray'

        resultado = aporte_SAIDI.loc[(aporte_SAIDI['LATITUD'] == row["LATITUD"]) & (aporte_SAIDI['LONGITUD'] == row["LONGITUD"]), 'escalada']

        # Comprobar si se encontró el resultado
        if not resultado.empty:
            radio = float(resultado.iloc[0]) 

        folium.CircleMarker(
            location=(row["LATITUD"], row["LONGITUD"]),
            radius=radio,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=1.0,
            popup=f"Trafo Fase: {row.PHASES} \n Propietario: {row.OWNER1} \n Impedancia: {row.IMPEDANCE} \n Marca: {row.MARCA} \n Fecha fabricacion: {row.DATE_FAB[:10]} \n Tipo subestación: {row.TIPO_SUB} \n KVA: {row.KVA} \n KV1: {row.KV1}"
        ).add_to(mapa)

    for _, row in apoyos_seleccionado.iterrows():
        point1 = (row["LATITUD"], row["LONGITUD"])
        radio = 5
        if (point1 in puntos_equipos):
            color = 'blue'
        else:
            color = 'gray'
        
        resultado = aporte_SAIDI.loc[(aporte_SAIDI['LATITUD'] == row["LATITUD"]) & (aporte_SAIDI['LONGITUD'] == row["LONGITUD"]), 'escalada']

        # Comprobar si se encontró el resultado
        if not resultado.empty:
            radio = float(resultado.iloc[0]) 

        folium.CircleMarker(
            location=(row["LATITUD"], row["LONGITUD"]),
            radius=radio,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=1.0,
            popup=f"Apoyo Propietario: {row.TOWNER} \n Tipo: {row.TIPO} \n Clase: {row.CLASE} \n Material: {row.MATERIAL} \n Longitud: {row.LONG_APOYO} \n Tierra pie: {row.TIERRA_PIE} \n Vientos: {row.VIENTOS}"
        ).add_to(mapa)

    for _, row in switches_seleccionado.iterrows():
        point1 = (row["LATITUD"], row["LONGITUD"])
        radio = 5
        if (point1 in puntos_equipos):
            color = 'purple'
        else:
            color = 'gray'

        resultado = aporte_SAIDI.loc[(aporte_SAIDI['LATITUD'] == row["LATITUD"]) & (aporte_SAIDI['LONGITUD'] == row["LONGITUD"]), 'escalada']

        # Comprobar si se encontró el resultado
        if not resultado.empty:
            radio = float(resultado.iloc[0]) 

        folium.CircleMarker(
            location=(row["LATITUD"], row["LONGITUD"]),
            radius=radio,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=1.0,
            popup=f"Switche Fase: {row.PHASES} \n Codigo assembly: {row.ASSEMBLY} \n KV: {row.KV} \n Estado: {row.STATE}"
        ).add_to(mapa)
    

    mapa_html = mapa._repr_html_()

    for idx, sample_idx in enumerate(top_3_indices):

        # Determinar el tipo de equipo según el índice
        if sample_idx >= a4_start:
            tipo_equipo = 'apoyo'
        else:
            tipo_equipo = equipo_nombres.get(a[sample_idx, 1], 'desconocido')  # Columna 1 determina el tipo

        # Obtener las 10 columnas más relevantes según `mask3`
        sample_idx_position = np.where(top_3_indices == sample_idx)[0][0]
        top_10_indices = np.argsort(mask3[sample_idx_position])[-10:][::-1]  # Ordenar en descendente
        top_10_columnas = [columns[i] for i in top_10_indices]

        # Graficar el gráfico interactivo
        grafo = graficar_grafo_interactivo_2(
            mask3[sample_idx_position][top_10_indices].reshape(1, -1),
            top_10_columnas,
            width=100,
            height=750
        )

        # Guardar el gráfico como archivo HTML
        grafo.write_html(f"./graficos_interactivos/grafo_muestra_{idx+1}.html")
        print(f"Archivo HTML generado: graficos_interactivos/grafo_muestra_{idx+1}.html (Tipo de Equipo: {tipo_equipo})")

    _,masks4=loaded_clf.explain(a)
    masks4=np.array(list(masks4.values())).sum(axis=0)
    aux_ind=[]
    aux_eq=[]
    # Iterar sobre las 3 muestras más relevantes
    for idx in range(masks4.shape[0]):
        # Determinar el tipo de equipo según el índice
        if idx>= a4_start:
            tipo_equipo = 'apoyo'  # Índices correspondientes a `a4`
        else:
            tipo_equipo = equipo_nombres.get(a[idx, 1], 'desconocido')  # Columna 1 determina el tipo
        aux_eq.append(tipo_equipo)

    grafo = graficar_grafo_interactivo_2(masks4.T,aux_eq,iteraciones=100,vector=y_e,width=100,height=750)
    # Guardar el gráfico como archivo HTML
    grafo.write_html(f"./graficos_interactivos/grafo_interactivo_0.html")    

    # Ruta del directorio con los archivos HTML generados
    output_dir = "./graficos_interactivos/"
    output_combined_file = "./graficos_interactivos/grafo_combinado.html"

    # Lista de archivos HTML generados
    html_files = [f"grafo_muestra_{i + 1}.html" for i in range(len(top_3_indices))] + ["grafo_interactivo_0.html"]

    # Iniciar el contenido del archivo combinado
    html_combined = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Gráficos Combinados</title>
        <style>
            body {
                font-family: Arial, sans-serif;
            }
            .container {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            iframe {
                margin: 20px;
                border: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
    """

    # Agregar cada archivo HTML como un iframe
    for idx, html_file in enumerate(html_files):
        file_path = os.path.join(output_dir, html_file)
        if os.path.exists(file_path):
            if idx == 3:
                html_combined += f"""
                    <h3>Grafos red afectada</h3>
                    <iframe src="{html_file}" width="100%" height="600px"></iframe>
                    """
            else:
                html_combined += f"""
                <h3>Grafo equipo {tipo_equipos_criticos[idx]} con ID {codigo_equipos_criticos[idx]}</h3>
                <iframe src="{html_file}" width="100%" height="600px"></iframe>
                """
        else:
            print(f"Advertencia: No se encontró el archivo {file_path}")

    # Cerrar el HTML combinado
    html_combined += """
        </div>
    </body>
    </html>
    """

    # Guardar el archivo combinado
    os.makedirs(output_dir, exist_ok=True)
    with open(output_combined_file, "w", encoding="utf-8") as f:
        f.write(html_combined)  

    # Abrir el archivo HTML en el navegador
    # Asegúrate de que la ruta sea absoluta
    ruta_absoluta = os.path.abspath("./graficos_interactivos/grafo_combinado.html")

    # Abre el archivo en el navegador
    webbrowser.open(f'file://{ruta_absoluta}')  
    

    return mapa_html
