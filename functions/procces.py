import pandas as pd 
from langchain_community.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaLLM
import pickle
import os




def eventos_transformadores_procces(query: str, model:str, chat_id:str) -> str:
    eventos_trafos = pd.read_csv('structured_data/Tabla_General.csv')
    NUMERIC_COLUMNS = eventos_trafos.select_dtypes(include=['number']).columns.tolist()
    CATEGORICAL_COLUMNS= eventos_trafos.select_dtypes(include=['object', 'category']).columns.tolist()

    eventos_trafos[NUMERIC_COLUMNS] = eventos_trafos[NUMERIC_COLUMNS].apply(pd.to_numeric, errors='coerce')

    # 4. Convertir columnas categóricas al tipo 'category'
    # Esto optimiza el uso de memoria y puede mejorar el rendimiento en ciertas operaciones
    eventos_trafos[CATEGORICAL_COLUMNS] = eventos_trafos[CATEGORICAL_COLUMNS].astype('category')


    # Supongamos que tu DataFrame se llama df y la columna de fecha se llama 'fecha_sin_hora'
    eventos_trafos['FECHA'] = pd.to_datetime(eventos_trafos['FECHA'].astype(str), format='%Y-%m-%d')
    # Supongamos que la columna de fecha y hora se llama 'fecha_con_hora'
    eventos_trafos['inicio'] = pd.to_datetime(eventos_trafos['inicio'].astype(str), format='%Y-%m-%d %H:%M:%S')
    eventos_trafos['fin'] = pd.to_datetime(eventos_trafos['fin'].astype(str), format='%Y-%m-%d %H:%M:%S')


    
    
    # if model=="gpt":
    #     chat_model=ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    # elif model=="llama3.1":
    #     chat_model=ChatOllama(model="llama3.1",temperature=0)
    # elif model=="llama3.2":
    #     chat_model=ChatOllama(model="llama3.2:1b",temperature=0)
    chat_model=ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    try:
        agent = create_pandas_dataframe_agent(
        chat_model,
        eventos_trafos,
        verbose=True,
        agent_type="openai-functions",
        allow_dangerous_code=True,
        include_df_in_prompt=True,  # Incluye las primeras filas del DataFrame en el prompt
        number_of_head_rows=5)
        response=agent.invoke(query)["output"]
    except:
        response="De acuerdo a mi conocimiento actual, no tengo la capacidad para responder a tu pregunta, por favor reformula tu pregunta."

    return response

def eventos_transformadores_plots_procces(query: str, model:str, chat_id:str) -> str:
    
    number_image=int(len(os.listdir(f"plots/{chat_id}")))
    path_plot=f"plots/{chat_id}/output_{number_image}.jpg"

    eventos_trafos = pd.read_csv('structured_data/Tabla_General.csv')
    NUMERIC_COLUMNS = eventos_trafos.select_dtypes(include=['number']).columns.tolist()
    CATEGORICAL_COLUMNS= eventos_trafos.select_dtypes(include=['object', 'category']).columns.tolist()

    eventos_trafos[NUMERIC_COLUMNS] = eventos_trafos[NUMERIC_COLUMNS].apply(pd.to_numeric, errors='coerce')

    # 4. Convertir columnas categóricas al tipo 'category'
    # Esto optimiza el uso de memoria y puede mejorar el rendimiento en ciertas operaciones
    eventos_trafos[CATEGORICAL_COLUMNS] = eventos_trafos[CATEGORICAL_COLUMNS].astype('category')


    # Supongamos que tu DataFrame se llama df y la columna de fecha se llama 'fecha_sin_hora'
    eventos_trafos['FECHA'] = pd.to_datetime(eventos_trafos['FECHA'].astype(str), format='%Y-%m-%d')
    # Supongamos que la columna de fecha y hora se llama 'fecha_con_hora'
    eventos_trafos['inicio'] = pd.to_datetime(eventos_trafos['inicio'].astype(str), format='%Y-%m-%d %H:%M:%S')
    eventos_trafos['fin'] = pd.to_datetime(eventos_trafos['fin'].astype(str), format='%Y-%m-%d %H:%M:%S')

    head_df = eventos_trafos.head(5).to_string(index=False)
    
    descripcion_df=""" 
    Este DataFrame contiene información acerca de interrupciones o eventos presentadas en redes eléctricas de media tensión, 
    más específicamente en tres tipos de equipos: Tranformadores, interruptores y tramos de linea (tramos de red).

    Las columnas incluyen: 
    - **Evento**: Id de la interrupción o el evento.
    - **equipo_ope**: Código del equipo en el que ocurrió la interrupción.
    - **tipo_equi_ope**: Me indica si la interrupción ocurrió sobre un Transformador, o sobre un interruptor o sobre un tramo de linea, es decir que tiene solo tres posibles valores.
    - **cto_equi_ope**: Código del circuito al que pertenece el equipo en el cual se dió la interrupción.
    - **tipo_elemento**: Capacidad en Kilo Voltios del equipo en el cual ocurrió la interrupción, tiene 4 posibles valores: 33, 13.2, TFD y TFP
    - **inicio**: Fecha y hora del inicio del evento o interrupción.
    - **fin**: Fecha y hora de la finalización del evento o interrupción.
    - **duracion_h**: Duración en horas del evento o interrupción.
    - **tipo_duracion**: Variable categórica que indica si ele vento duró más de tres minutos o no; por tanto, tiene dos posibles valores: > 3 min y <= 3 min
    - **causa**: Causa del evento o interrupción.
    - **CNT_TRAFOS_AFEC**: Cantidad de transformadores afectados en la interrupción o evento.
    - **cnt_usus**: Cantidad de usuarios afectados por la interrupción o evento.
    - **SAIDI**: Indicador que mide el promedio de la duración en horas de la interrupción por usuario.
    - **SAIFI**: Indicador que mide el promedio de cantidad de interrupciones por usuario.
    - **PHASES**: Número de fases del equipo en el que ocurrió la interrupción; por tanto tiene 3 posibles valores: 3., 1., 2.
    - **FPARENT**: Código del circuito que contiene el equipo en donde se presentó la interrupción.
    - **FECHA**: Fecha en la que se presentó el evento o interrupción.
    - **LONGITUD**: Longitud geográfica de la ubicación del equipo en el que se presentó la interrupción o evento.
    - **LATITUD**: Latiud geográfica de la ubicación del equipo en el que se presentó la interrupción o evento.
    - **DEP**: Departamento en donde se presentó la interrupción o evento.
    - **MUN**: Municipio en donde se presentó la interrupción o evento.

    A continuación, se muestran las primeras 5 filas del DataFrame:

   {head_df}
    """

    suffix_instrucciones ="""
    Construye el gráfico de la forma más estética posible para mostrar a un usuario. 
    Puedes utilizar los siguientes colores: verde y gris en diferentes tonalidades (si es necesario, utiliza más colores).
    Además, los títulos y ejes de los gráficos deben estar en español.
    Guarda la imagen en la ruta relativa {path_plot}.
    No ejecutes el comando plt.show().
    Siempre ejecuta el comando plt.tight_layout()


    Ojo, la información que utilizes para el gráfico y las conclusiones debe ser extraida únicamente del DataFrame proporcionado, no inventes.
    """
    
    query=f"""{query}. Construye el gráfico de la forma más estética posible para mostrar a un usuario. 
    Puedes utilizar los siguientes colores: verde y gris en diferentes tonalidades (si es necesario, utiliza más colores).
    Además, los títulos y ejes de los gráficos deben estar en español.
    Guarda la imagen en la ruta relativa {path_plot}.
    No ejecutes el comando plt.show().
    Siempre ejecuta el comando plt.tight_layout().
    Siempre poner los ejes del gráfico con un fontsize = 14.
    Siempre rotar las etiquetas del eje x a 90 grados.
    """
    # Al final, redacta conclusiones basadas **únicamente en los datos proporcionados en el DataFrame**.
    # Asegúrate de que todas las estadísticas y observaciones estén directamente derivadas de los datos.

    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), eventos_trafos, verbose=True, allow_dangerous_code=True)
    
    try:
        agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        eventos_trafos,
        verbose=True, 
        agent_type="openai-functions",#prefix=descripcion_df,  # Añade la descripción al inicio del prompt #suffix=suffix_instrucciones.format(path_plot=path_plot),
        allow_dangerous_code=True,
        include_df_in_prompt=True,  # Incluye las primeras filas del DataFrame en el prompt
        number_of_head_rows=5)

        response=agent.invoke(query)["output"]
        flag_image=True
    except:
        response="De acuerdo a mi conocimiento actual, no tengo la capacidad para responder a tu pregunta, por favor reformula tu pregunta."
        flag_image=False

    with open("answer.pkl", 'wb') as archivo:
        pickle.dump(response, archivo)
    
    return response, flag_image