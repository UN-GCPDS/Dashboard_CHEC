import os
import json
import shutil
import random
import datetime
import operator
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import date
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy.special import softmax
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
import math




def create_combined_visualizations(mask, y_categorized, columns, SAIDI):
    # Crear una figura grande para contener todas las visualizaciones
    fig = plt.figure(figsize=(25, 50))

    # Definir la disposición de los subplots
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1,1], hspace=0.3)

    # 1. Visualización de la máscara original
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(mask, aspect='auto', cmap='viridis')
    ax1.set_title("Relevancia", fontsize=14)
    ax1.set_xlabel("Características", fontsize=12)
    ax1.set_ylabel("Muestras", fontsize=12)
    cbar = fig.colorbar(im, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label("Valores", fontsize=12)

    # 2. Gráficos de violín
    ax2 = fig.add_subplot(gs[1])
    normalized_mask = softmax(mask, axis=1)
    column_relevance = np.mean(normalized_mask, axis=0)
    top_20_indices = np.argsort(column_relevance)[-20:]
    top_20_columns = columns[top_20_indices]

    gs2 = gs[1].subgridspec(1, 5, wspace=0.3)
    axes_violin = [fig.add_subplot(gs2[0, i]) for i in range(5)]

    class_titles = ['todas las muestras', 'Riesgo Bajo', 'Riesgo Medio', 'Riesgo Alto','Riesgo Muy Alto']
    class_indices = [0, 1, 2,3]
    colors = sns.color_palette("muted", len(top_20_columns))

    # Gráfica para todas las muestras
    top_20_mask = mask[:, top_20_indices]
    sns.violinplot(data=top_20_mask, inner="box", cut=0, palette=colors, ax=axes_violin[0])
    axes_violin[0].set_xticks(range(len(top_20_columns)))
    axes_violin[0].set_xticklabels(top_20_columns, rotation=90, fontsize=10)
    axes_violin[0].set_ylabel("Valores", fontsize=12)
    axes_violin[0].set_title(class_titles[0], fontsize=14)

    # Gráficas por clase
    for i, class_idx in enumerate(class_indices, 1):
        if np.any(y_categorized == class_idx):
            column_relevance = np.mean(mask[y_categorized == class_idx], axis=0)
            top_20_indices = np.argsort(column_relevance)[-20:]
            top_20_columns = columns[top_20_indices]
            top_20_mask = mask[:, top_20_indices]
            sns.violinplot(data=top_20_mask, inner="box", cut=0, palette=colors, ax=axes_violin[i])
            axes_violin[i].set_xticks(range(len(top_20_columns)))
            axes_violin[i].set_xticklabels(top_20_columns, rotation=90, fontsize=10)
            axes_violin[i].set_title(class_titles[i], fontsize=14)
        else:
            axes_violin[i].text(0.5, 0.5, f'No hay datos para {class_titles[i]}', fontsize=14, ha='center')
            axes_violin[i].set_axis_off()

    # 3. Gráficos de barras
    gs3 = gs[2].subgridspec(1, 5, wspace=0.3)
    axes_bar = [fig.add_subplot(gs3[0, i]) for i in range(5)]

    # Gráfica para todas las muestras
    top_20_relevance = column_relevance[top_20_indices]
    axes_bar[0].bar(top_20_columns, top_20_relevance, color=sns.color_palette("muted", len(top_20_columns)))
    axes_bar[0].set_xticks(range(len(top_20_columns)))
    axes_bar[0].set_xticklabels(top_20_columns, rotation=90, fontsize=10)
    axes_bar[0].set_ylabel("Valores", fontsize=12)
    axes_bar[0].set_title("Todas las muestras", fontsize=14)

    class_titles = ["Riesgo Bajo", "Riesgo Medio", "Riesgo Alto",'Riesgo Muy Alto']
    for i, class_idx in enumerate([0, 1, 2,3]):
        ax_index = i + 1
        if np.any(y_categorized == class_idx):
            column_relevance = np.mean(mask[y_categorized == class_idx], axis=0)
            top_20_indices = np.argsort(column_relevance)[-20:]
            top_20_columns = columns[top_20_indices]
            top_20_relevance = column_relevance[top_20_indices]
            axes_bar[ax_index].bar(top_20_columns, top_20_relevance, color=sns.color_palette("muted", len(top_20_columns)))
            axes_bar[ax_index].set_xticks(range(len(top_20_columns)))
            axes_bar[ax_index].set_xticklabels(top_20_columns, rotation=90, fontsize=10)
            axes_bar[ax_index].set_title(class_titles[i], fontsize=14)
        else:
            axes_bar[ax_index].text(0.5, 0.5, f'No hay datos para {class_titles[i]}', fontsize=14, ha='center')
            axes_bar[ax_index].set_axis_off()

    # 4. Mapas de calor de correlación
    mask_df = pd.DataFrame(normalized_mask, columns=columns)
    top_20_columns_all = calculate_top_20_columns(normalized_mask, columns)
    top_20_mask_df = mask_df[top_20_columns_all]

    gs4 = gs[3].subgridspec(1, 5, wspace=0.3)
    axes_heatmap = [fig.add_subplot(gs4[0, i]) for i in range(5)]

    groups = {
        "Total": top_20_mask_df,
        "Riesgo Bajo": top_20_mask_df[y_categorized == 0],
        "Riesgo Medio": top_20_mask_df[y_categorized == 1],
        "Riesgo Alto": top_20_mask_df[y_categorized == 2],
        "Riesgo Muy Alto": top_20_mask_df[y_categorized == 3],
    }

    for group_name, group_data in groups.items():
        if 'SAIDI' not in group_data.columns:
            groups[group_name] = group_data.copy()
            groups[group_name]['SAIDI'] = SAIDI[group_data.index]

    for i, (group_name, group_data) in enumerate(groups.items()):
        top_20_corr = top_20_correlation(group_data)

        if top_20_corr is not None:
            sns.heatmap(
                top_20_corr,
                cmap="coolwarm",
                ax=axes_heatmap[i],
                cbar=i==3,
                fmt=".2f",
                vmin=-1,
                vmax=1
            )
            axes_heatmap[i].set_title(f"{group_name}", fontsize=14)
            axes_heatmap[i].tick_params(axis='x', rotation=90, labelsize=10)
            axes_heatmap[i].tick_params(axis='y', labelsize=10)
        else:
            axes_heatmap[i].text(0.5, 0.5, 'No hay datos disponibles',
                               horizontalalignment='center',
                               verticalalignment='center',
                               fontsize=16,
                               color='red')
            axes_heatmap[i].set_title(f"{group_name}", fontsize=14)
            axes_heatmap[i].set_axis_off()

    # Ajustar el diseño final
    plt.tight_layout()

    return fig


# Funciones auxiliares
def calculate_top_20_columns(mask, columns, top_n=20):
    column_relevance = np.mean(mask, axis=0)
    top_20_indices = np.argsort(column_relevance)[-top_n:]
    return columns[top_20_indices]

def top_20_correlation(data, reference_col='SAIDI'):
    if data.empty:
        return None
    correlation_matrix = data.corr()
    correlation_with_reference = correlation_matrix.abs()
    return correlation_with_reference


def load_data():

    mask = np.load(os.path.join("..", "data", "mask.npy"))
    
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

    Xdata = pd.read_pickle(os.path.join("..", "data", "SuperEventos_Criticidad_AguasAbajo_CODEs.pkl"))
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

    # Preparar datos
    y = y1.astype('float32')
    # Crear categorías basadas en percentiles
    percentiles = np.percentile(y[:, 0], [33.33, 66.66])
    y_categorized = np.digitize(y[:, 0:1].flatten(), bins=percentiles).astype(int)

    # Crear la lista de opciones
    options = [{"label": " ", "value": ""}] + [{"label": col, "value": col} for col in df1.columns.to_list()]

    if os.path.exists("./options/criterias_2.json"):
        os.remove("./options/criterias_2.json")

    # Guardar en un archivo JSON
    with open("./options/criterias_2.json", "w") as f:
        json.dump(options, f)


    return mask, trafos, apoyos, redmt, switches, Xdata, df1, y, y_categorized, SAIDI


def graphics_PDF(criterias, data_frame_select, mask, Xdata, y, y_categorized, SAIDI, count):

    if count == 0:

        # Verifica si la carpeta existe
        if not os.path.exists('./outputs_PDFs'):
            print(f"La carpeta {'./outputs_PDFs'} no existe.")
            return
        
        # Itera sobre todos los elementos dentro de la carpeta
        for elemento in os.listdir('./outputs_PDFs'):
            ruta_elemento = os.path.join('./outputs_PDFs', elemento)
            try:
                # Si es un archivo, lo elimina
                if os.path.isfile(ruta_elemento) or os.path.islink(ruta_elemento):
                    os.remove(ruta_elemento)
                # Si es una carpeta, elimina la carpeta completa
                elif os.path.isdir(ruta_elemento):
                    shutil.rmtree(ruta_elemento)
            except Exception as e:
                print(f"No se pudo eliminar {ruta_elemento}. Error: {e}")

    for i in range(0,4):
        
        if (criterias[i][0] != '') and (criterias[i][1] != ''):

            match criterias[i][0]:

                case 'seleccion':

                    data_frame_select = data_frame_select[data_frame_select[criterias[i][1]] == criterias[i][2]]

                case 'rango_num':

                    operators = {
                                    '<': operator.lt,
                                    '>': operator.gt,
                                    '==': operator.eq,
                                    '<=': operator.le,
                                    '>=': operator.ge,
                                    '!=': operator.ne
                                }
                    
                    data_frame_select = data_frame_select[operators[criterias[i][2]](data_frame_select[criterias[i][1]], float(criterias[i][3]))]

                case 'fecha':

                    data_frame_select = data_frame_select[(data_frame_select[criterias[i][1]] >= criterias[i][2]) & (data_frame_select[criterias[i][1]] <= criterias[i][3])]

                case _:

                    None

    # Crear las imágenes modificadas
    ind=data_frame_select.index
    mask=mask[ind]
    mask_df=pd.DataFrame(mask)
    columns = Xdata.columns
    mask_df.columns=columns
    mask_df['SAIDI'] = y[ind, 0]
    y_categorized=y_categorized[ind]
    SAIDI=SAIDI[ind]
    fig = create_combined_visualizations(mask, y_categorized, columns, SAIDI)
    plt.savefig(f'./outputs_PDFs/graphics_criticality_{count}.pdf', bbox_inches='tight', dpi=300)
    plt.close()