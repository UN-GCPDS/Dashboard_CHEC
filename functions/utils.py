# functions/utils.py
import json
import os
import pickle
import webbrowser
import pandas as pd
from neo4j import GraphDatabase
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain.document_loaders import PyPDFLoader #To load pdf files
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter #To splitt the text

import pandas as pd 
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import OpenAI
import time
import shutil 
from pyvis.network import Network
import httpx


import maps_page

from functions.procces import eventos_transformadores_procces,eventos_transformadores_plots_procces
from functions.tools import capitulo_1, capitulo_2, capitulo_3, capitulo_4, resolucion_40117,eventos_transformadores, eventos_transformadores_plots, normativa_apoyos, normativa_protecciones, normativa_aisladores, redes_aereas_media_tension, codigo_electrico_colombiano, requisitos_redes_aereas, retie


http_client = httpx.Client(verify=False)
http_async_client = httpx.AsyncClient(verify=False)

def update_documents_procces(name_procces,path_uploaded_pdf):
    path_vectorial_database=f"embeddings_by_procces/{name_procces}"
    if (os.path.exists(path_vectorial_database) and os.path.isdir(path_vectorial_database)):
        shutil.rmtree(path_vectorial_database)

    concatenated_files=[] #In this list we will storage all the content and the metadata of the unstructured loaded files
    loader = PyPDFLoader(path_uploaded_pdf) #Load the file
    data = loader.load() #Carry the file to the type of object Document: List in the which for evey page we have a object with two atributes: metadata and page_content
    concatenated_files.extend(data)



    #As we can see, the LLM can procces a limit amount of tokens, so that we have to split the text in fragments of 1500 tokens in this case (because is the maximun amount of tokens that support our model)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=7500, #Fragments of text of 1500 tokens
        chunk_overlap=200, #For evey fragment that take the 200 last tokens of the last fragment
        length_function=len
        )

    documents = text_splitter.split_documents(concatenated_files) #List with the metadata and the content splitt by fragments of 1500 tokens

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") #word2vec model of openAI


    NOMBRE_INDICE_CHROMA = path_vectorial_database #Name of my vectorial database (Put the name that you want)

    #Creating our vectorial database or vector store
    vectorstore_chroma = Chroma.from_documents(
        documents=documents, #Create the database with the list of the created documents (Every instance will be the embedding of every document)
        embedding=embeddings, #Word2vec model to create our embeddings, always use the same.
        persist_directory=NOMBRE_INDICE_CHROMA #Load my database in the indicated folder (If I close the section, I will keep storaged my vectorial databas in the folder called "NOMBRE_INDICE_CHROMA" )
    )

def load_structured_data():
    eventos_trafos = pd.read_csv('structured_data/EVENTOS_TRAFOS.csv')

    eventos_trafos['inicio'] = pd.to_datetime(eventos_trafos['inicio']).dt.normalize()
    eventos_trafos['fin'] = pd.to_datetime(eventos_trafos['fin']).dt.normalize()
    eventos_trafos['DATE_FAB'] = pd.to_datetime(eventos_trafos['DATE_FAB'], format='%Y-%m-%d', errors='coerce').dt.normalize()
    eventos_trafos['inicio_m'] = pd.to_datetime(eventos_trafos['inicio_m']).dt.to_period('M')
    eventos_trafos['fin_m'] = pd.to_datetime(eventos_trafos['fin_m']).dt.to_period('M')
    eventos_trafos['FECHA'] = pd.to_datetime(eventos_trafos['FECHA']).dt.to_period('M')
    eventos_trafos['FECHA_ACT'] = pd.to_datetime(eventos_trafos['FECHA_ACT'], format='%Y-%m-%d', errors='coerce').dt.normalize()
    eventos_trafos[['duracion_h','CNT_TRAFOS_AFEC','cnt_usus','SAIDI','SAIFI','PHASES','XPOS','YPOS','Z*','R','G','B','IMPEDANCE*','GRUPO015','KVA','KV1']] = eventos_trafos[['duracion_h','CNT_TRAFOS_AFEC','cnt_usus','SAIDI','SAIFI','PHASES','XPOS','YPOS','Z*','R','G','B','IMPEDANCE*','GRUPO015','KVA','KV1']].astype('float32')
    eventos_trafos[['evento','equipo_ope','tipo_equi_ope','cto_equi_ope','CODE','FPARENT*','TRFTYPE','ELNODE','CASO*']] = eventos_trafos[['evento','equipo_ope','tipo_equi_ope','cto_equi_ope','CODE','FPARENT*','TRFTYPE','ELNODE','CASO*']].astype('string')

    return eventos_trafos

CHAT_DATA_FILE = 'chat_data.json'

def load_previous_conversations():
    if os.path.exists(CHAT_DATA_FILE):
        try:
            with open(CHAT_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
            
        except json.JSONDecodeError:
            # Si el archivo está corrupto, retorna estructura vacía
            return {'chats': {}, 'current_chat_id': None}
    else:
        return {'chats': {}, 'current_chat_id': None}

def save_conversations(data):
    print("Guardando datos en el archivo JSON...")
    with open(CHAT_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



def conversation(chat_id,query,model,procces):

    path_images=f"plots/{chat_id}"
    # Verifica si el directorio no existe
    if not os.path.exists(path_images):
        os.makedirs(path_images)
    else:
        pass

    if procces!="general":

        if procces=="interrrupciones_transformadores":
            response=eventos_transformadores_procces(query,model,chat_id)
            return response, False
        elif procces=="generate_plots":
            response, flag_image=eventos_transformadores_plots_procces(query,model,chat_id)
            return response, flag_image

        else:
            #PROMP TEMPLATE:
            template =  """Se te proporcionará una serie de textos que contienen instrucciones sobre cómo 
                        resolver preguntas acerca de normativas en redes eléctricas de nivel de tensión 2. Según estos textos,
                        responde a la pregunta de la manera más completa posible.

                        Dado el siguiente contexto y teniendo en cuenta el historial de la conversación, 
                        responde a las preguntas hechas por el usuario:

                        {context}

                        {chat_history}
                        Human: {human_input}
                        Chatbot (RESPUESTA FORMAL):"""
                        
            prompt = PromptTemplate(
                input_variables=["chat_history", "human_input", "context"], template=template
            )

            memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

            if model=="llama3.1":
                chain = load_qa_chain(ChatOllama(model="llama3.1",temperature=0), chain_type="stuff", memory=memory, prompt=prompt)
            elif model=="llama3.2":
                chain = load_qa_chain(ChatOllama(model="llama3.2:1b",temperature=0), chain_type="stuff", memory=memory, prompt=prompt)
            elif model=="gpt":
                try:
                    # Primero intenta con el cliente asíncrono
                    chain = load_qa_chain(
                        ChatOpenAI(
                            model_name="gpt-3.5-turbo", 
                            temperature=0
                        ), 
                        chain_type="stuff", 
                        memory=memory, 
                        prompt=prompt
                    )
                except TypeError:
                    # Si falla, intenta con el cliente síncrono
                    chain = load_qa_chain(
                        ChatOpenAI(
                            model_name="gpt-3.5-turbo", 
                            temperature=0
                        ), 
                        chain_type="stuff", 
                        memory=memory, 
                        prompt=prompt
                    )
            
            # Load the chat history of the conversation for every particular agent
            path_memory=f"memories/{chat_id}.pkl"
            if os.path.exists(path_memory):
                with open(path_memory, 'rb') as f:
                    memory = pickle.load(f) #memory of the conversation
            
                chain.memory=memory

            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") #word2vec model of openAI
            # load from disk
            vectorstore = Chroma(persist_directory=f"embeddings_by_procces/{procces}",embedding_function=embeddings)

            docs=vectorstore.similarity_search(query,k=5) #Retriever

            print(docs)

            response=chain({"input_documents": docs, "human_input": query, "chat_history":memory}, return_only_outputs=False)['output_text'] #AI answer

            #Save the chat history (memory) for a new iteration of the conversation for the general agent:
            with open(path_memory, 'wb') as f:
                pickle.dump(chain.memory, f)

            return response, False


    elif procces == 'recomendacion':

        # Cargar la variable desde el archivo JSON 
        with open('./body_recommendations/'+str(chat_id)+'.json', 'r') as json_file: 
            data_equipo = json.load(json_file)

        if model=="llama3.2":

            model="llama2"
            
        elif model=="llama3.1":

            model="llama1"

            a = 1

        response = recomendacion(model, chat_id, data_equipo, query)
            
        return response, False

    else:
        # Abrir el archivo en modo de escritura
        with open(f"number_iteration.pkl", "w") as archivo:
            # Escribir el número entero en el archivo
            number_iteration = 0
            archivo.write(str(number_iteration))
        
        tools=[capitulo_1,capitulo_2,capitulo_3,capitulo_4,resolucion_40117,normativa_apoyos,normativa_protecciones,normativa_aisladores, redes_aereas_media_tension, codigo_electrico_colombiano, requisitos_redes_aereas, retie,eventos_transformadores,eventos_transformadores_plots]
        # if model=="gpt":
        #     llm_agent=ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        # elif model=="llama3.1":
        #     llm_agent=ChatOllama(model="llama3.1",temperature=0)
        # elif model=="llama3.2":
        #     llm_agent=ChatOllama(model="llama3.2:1b",temperature=0)
        try:
            llm_agent=ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        except:
            llm_agent=ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        # Prompt to use in the LangChain Agent:
        prompt = hub.pull("hwchase17/openai-functions-agent")
        prompt.messages[0].prompt.template="""Eres un asistente de servicio al cliente artificial que tiene una conversación con un asistente de servicio al cliente humano y tu función es responder 
        las preguntas hechas por el humano. Dada una pregunta hecha por el humano, elije que tool invocar. Si la pregunta no está relacionada con la información de ninguno de los tools, 
        proporcionale información al usuario a cerca de qué puede realizar preguntas, teniendo en cuenta el tipo de preguntas que puede responder cada uno de los tools."""      #"Eres un asistente de  que dada una pregunta de un usuario, elijirá que tool invocar. Si te preguntan que quien eres, responde: Soy una IA de experiencia memorable del cliente. Tienes conocimiento a cerca de los siguientes documentos: [Información General a cerca de EFIGAS S.pdf, INSTRUCTIVO GESTION DE SOLICITUDES QUEJAS Y RECLAMOS CON APLICABILIDAD IA.pdf, ANS.pdf, manual-de-comunicacion-interna.pdf, Politica de SAC.pdf, INSTRUCTIVO DE GESTION DE SOLICITUDES PROCESO DE SERVICIO AL CLIENTE  V16 CEREBRO EFIGAS.pdf, IN-SC-02_INSTRUCTIVO DE RECEPCION DE LLAMADAS DE CALL CENTER (V4) FINAL.pdf] .Si te hacen múltiples preguntas a la vez y es necesario invocar varios tools, hazlo, pero a cada tool pasale solo la porción de la pregunta que le corresponde. Al final, la respuesta que debes entregar es la unión  de las respuestas retornadas por cada tool (Literales, sin cambiarles nada), de una manera coherente y no redundante."
        agent = create_openai_functions_agent(llm_agent, tools, prompt) #Create the LangChain Agent
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) #Create the LangChain Agent Executor
        
        
        print(query,model,chat_id)
        print("MODEL_OUT")
        print(model)
        if model=="llama3.2":
            model="llama2"
        elif model=="llama3.1":
            model="llama1"
        info_execution=agent_executor.invoke({"input":[query,model,str(chat_id)]})
        
        
        # # Load the chat history of the conversation (AI MEssage and HumanMessage)
        # with open(f'memories/{chat_id}.pkl', 'rb') as f:
        #     conversation_history = pickle.load(f) #memory of the conversation
        
        
        # Abrir el archivo en modo de lectura
        with open(f"number_iteration.pkl", "r") as archivo:
            # Leer el contenido del archivo
            contenido = archivo.read()
            # Convertir el contenido a un número entero
            number_iteration = int(contenido)
            
        if number_iteration==0:
            response=info_execution["output"]
            flag_image=False
            
        elif number_iteration==3:
            with open("answer.pkl", 'rb') as archivo:
                response = pickle.load(archivo)
            flag_image=True
        elif number_iteration==4:
            with open("answer.pkl", 'rb') as archivo:
                response = pickle.load(archivo)
            flag_image=False

        else:
            with open("answer.pkl", 'rb') as archivo:
                response = pickle.load(archivo)
            flag_image=False
            

        print("Number Iteration")
        print(number_iteration)
        
        
        return response, flag_image

def get_structure_graph_recomendation(info_poligono):
    variables_por_equipo={}
    normativas_por_equipo={}
    documentos_por_equipo={}
    sugerencias_por_equipo={}
    ids_equipos=list(info_poligono.keys())
    for id in ids_equipos:
        tipo_equipo=info_poligono[id]["Tipo_de_equipo"]
        variables_recomendacion=pd.read_excel(os.path.join("..", "data", "arbol_decision_recomendaciones", f"variables_{tipo_equipo}.xlsx"))
        variables_por_equipo[tipo_equipo+"_"+id]=[]
        normativas_por_equipo[tipo_equipo+"_"+id]={}
        documentos_por_equipo[tipo_equipo+"_"+id]={}
        sugerencias_por_equipo[tipo_equipo+"_"+id]={}
        variables=list(info_poligono[id]["top_5"].keys())
        variables_id = [elemento + "_"+ id for elemento in variables]
        for i,var in enumerate(variables):
            valor=info_poligono[id]["top_5"][var]
            if var in list(variables_recomendacion["Variables"]):
                pass
            else:
                variable_modificada = verifify_subcadena(var, 'pres','Presión Atmosférica')
                if variable_modificada == var:
                    variable_modificada = verifify_subcadena(var, 'rh','Humedad Relativa')
                    if variable_modificada == var:
                        variable_modificada = verifify_subcadena(var, 'slp','Presión a Nivel del Mar')
                        if variable_modificada == var:
                            variable_modificada = verifify_subcadena(var, 'solar_rad','Radiación Solar')
                            if variable_modificada == var:
                                variable_modificada = verifify_subcadena(var, 'temp','Temperatura Ambiente')
                                if variable_modificada == var:
                                    variable_modificada = verifify_subcadena(var, 'uv','Índice UV')
                                    if variable_modificada == var:
                                        variable_modificada = verifify_subcadena(var, 'vis','Visibilidad')
                                        if variable_modificada == var:
                                            variable_modificada = verifify_subcadena(var, 'wind_gust_spd','Ráfagas de Viento')
                                            if variable_modificada == var:
                                                variable_modificada = verifify_subcadena(var, 'wind_spd','Velocidad Promedio del Viento')
                                            else:
                                                var = variable_modificada
                                        else:
                                            var = variable_modificada
                                    else:
                                        var = variable_modificada
                                else:
                                    var = variable_modificada
                            else:
                                var = variable_modificada
                        else:
                            var = variable_modificada
                    else:
                        var = variable_modificada
                else:
                    var = variable_modificada
            print(variables_recomendacion[variables_recomendacion["Variables"]==var])
            if not variables_recomendacion[variables_recomendacion["Variables"]==var].empty:
                try:
                    documento_buscar=variables_recomendacion[variables_recomendacion["Variables"]==var]["Documento "].iloc[0]
                except:
                    documento_buscar=variables_recomendacion[variables_recomendacion["Variables"]==var]["Documento"].iloc[0]
            else:
                continue
            sugerencia=variables_recomendacion[variables_recomendacion["Variables"]==var]["Sugerencia"].iloc[0]
            seccion_buscar=variables_recomendacion[variables_recomendacion["Variables"]==var]["Normativa"].iloc[0]
            description_var=variables_recomendacion[variables_recomendacion["Variables"]==var]["Descripción"].iloc[0]
            
            variables_por_equipo[tipo_equipo+"_"+id].append({"nombre":variables_id[i],"descripcion":description_var,"valor":valor})
            normativas_por_equipo[tipo_equipo+"_"+id][variables_id[i]]={"id": "norm"+var+"_"+id, "descripcion": seccion_buscar}
            documentos_por_equipo[tipo_equipo+"_"+id][variables_id[i]]={"id": "doc"+var+"_"+id, "nombre": documento_buscar}
            sugerencias_por_equipo[tipo_equipo+"_"+id][variables_id[i]]={"id": "sug"+var+"_"+id, "descripcion": sugerencia}

    return variables_por_equipo, normativas_por_equipo, documentos_por_equipo, sugerencias_por_equipo



def create_node(tx, label, properties):
    # Excluir el 'id' del mapeo SET para evitar sobrescribirlo
    properties_without_id = {k: v for k, v in properties.items() if k != 'id'}
    
    # Crear la parte del SET con formato "clave: $clave"
    set_clause = ', '.join([f"{k}: ${k}" for k in properties_without_id.keys()])
    
    query = (
        f"MERGE (n:{label} {{id: $id}}) "
        f"SET n += {{{set_clause}}} "
        f"RETURN n"
    )
    
    tx.run(query, **properties)

def create_relationship(tx, label1, id1, relation, label2, id2):
    query = (
        f"MATCH (a:{label1} {{id: $id1}}), (b:{label2} {{id: $id2}}) "
        f"MERGE (a)-[:{relation}]->(b)"
    )
    tx.run(query, id1=id1, id2=id2)
def get_html_graph_recomendation(variables_por_equipo,normativas_por_equipo,documentos_por_equipo,sugerencias_por_equipo,chat_id):
    # --- Configuración de Neo4j ---
    NEO4J_URI = "bolt://localhost:7687"  # Reemplaza con tu URI
    NEO4J_USER = "neo4j"                # Reemplaza con tu usuario
    NEO4J_PASSWORD = "proyectochec"         # Reemplaza con tu contraseña

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


    # Inicializar la red PyVis con fondo blanco y texto en negro
    net = Network(height="750px", width="100%", bgcolor="white", font_color="black")

    # --- Definición de paleta de colores ---
    color_equipo = "#4caf50"            # Verde medio para equipos
    color_variable = "#9e9e9e"          # Gris para variables
    color_normativa = "#2e7d32"         # Verde oscuro para normativas
    color_documento = "#757575"         # Gris medio oscuro para documentos
    color_sugerencia = "#81c784"        # Verde claro para sugerencias

    with driver.session() as session:
        for equipo_id, vars_lista in variables_por_equipo.items():
            # Agregar nodo de equipo con color verde en PyVis
            net.add_node(equipo_id, label=equipo_id, shape="ellipse", color=color_equipo, 
                        title=f"Equipo: {equipo_id}")

            # Crear nodo de equipo en Neo4j
            equipo_properties = {
                "id": equipo_id,
                "label": equipo_id,
                "shape": "ellipse",
                "color": color_equipo,
                "title": f"Equipo: {equipo_id}"
            }
            session.write_transaction(create_node, "Equipo", equipo_properties)
            
            for var in vars_lista:
                var_nombre = var["nombre"]
                
                # Agregar nodo de variable en PyVis
                nodos_actuales = [n["id"] for n in net.nodes]
                if var_nombre not in nodos_actuales:
                    net.add_node(var_nombre, label=var_nombre, shape="box", color=color_variable,
                                title=f"{var['descripcion']}\nValor: {var['valor']}")
                
                # Relacionar equipo con variable en PyVis
                net.add_edge(equipo_id, var_nombre, label="Tiene Variable Crítica", title="Tiene Variable Crítica")
                
                # Crear nodo de variable en Neo4j
                variable_properties = {
                    "id": var_nombre,
                    "label": var_nombre,
                    "shape": "box",
                    "color": color_variable,
                    "description": var['descripcion'],
                    "value": var['valor']
                }
                session.write_transaction(create_node, "Variable", variable_properties)
                
                # Crear relación "TIENE_VARIABLE_CRITICA" entre Equipo y Variable en Neo4j
                session.write_transaction(create_relationship, "Equipo", equipo_id, "TIENE_VARIABLE_CRITICA", "Variable", var_nombre)
        
                # Asociar normativa
                norm_info = normativas_por_equipo.get(equipo_id, {}).get(var_nombre)
                if norm_info:
                    norm_id = norm_info["id"]
                    # Agregar nodo de normativa en PyVis si no existe
                    if norm_id not in [n["id"] for n in net.nodes]:
                        net.add_node(norm_id, label="Normativa", shape="diamond", color=color_normativa, 
                                    title=norm_info["descripcion"])
                    
                    # Agregar relación en PyVis
                    net.add_edge(var_nombre, norm_id, title="Regulado por", label="Regulado por")
                    
                    # Crear nodo de normativa en Neo4j
                    normativa_properties = {
                        "id": norm_id,
                        "descripcion": norm_info["descripcion"]
                    }
                    session.write_transaction(create_node, "Normativa", normativa_properties)
                    
                    # Crear relación "REGULADO_POR" entre Variable y Normativa en Neo4j
                    session.write_transaction(create_relationship, "Variable", var_nombre, "REGULADO_POR", "Normativa", norm_id)
        
                    # Asociar documento a la normativa
                    doc_info = documentos_por_equipo.get(equipo_id, {}).get(var_nombre)
                    if doc_info:
                        doc_id = doc_info["id"]
                        # Agregar nodo de documento en PyVis si no existe
                        if doc_id not in [n["id"] for n in net.nodes]:
                            net.add_node(doc_id, label=doc_info["nombre"], shape="database", color=color_documento, 
                                        title=f"Documento: {doc_info['nombre']}")
                        # Agregar relación en PyVis
                        net.add_edge(norm_id, doc_id, title="Documentado en", label="Documentado en")
                        
                        # Crear nodo de documento en Neo4j
                        documento_properties = {
                            "id": doc_id,
                            "nombre": doc_info["nombre"]
                        }
                        session.write_transaction(create_node, "Documento", documento_properties)
                        
                        # Crear relación "DOCUMENTADO_EN" entre Normativa y Documento en Neo4j
                        session.write_transaction(create_relationship, "Normativa", norm_id, "DOCUMENTADO_EN", "Documento", doc_id)
                
                # Asociar sugerencia a la variable
                sug_info = sugerencias_por_equipo.get(equipo_id, {}).get(var_nombre)
                if sug_info:
                    sug_id = sug_info["id"]
                    # Agregar nodo de sugerencia en PyVis si no existe
                    if sug_id not in [n["id"] for n in net.nodes]:
                        net.add_node(sug_id, label="Sugerencia", shape="star", color=color_sugerencia, 
                                    title=sug_info["descripcion"])
                    # Agregar relación en PyVis
                    net.add_edge(var_nombre, sug_id, title="con sugerencia", label="con sugerencia")
                    
                    # Crear nodo de sugerencia en Neo4j
                    sugerencia_properties = {
                        "id": sug_id,
                        "descripcion": sug_info["descripcion"]
                    }
                    session.write_transaction(create_node, "Sugerencia", sugerencia_properties)
                    
                    # Crear relación "TIENE_SUGERENCIA" entre Variable y Sugerencia en Neo4j
                    session.write_transaction(create_relationship, "Variable", var_nombre, "TIENE_SUGERENCIA", "Sugerencia", sug_id)

    # --- Configuración de física y opciones ---
    net.force_atlas_2based()  # Opcional, para un layout específico
    net.set_options("""
    var options = {
    "nodes": {
        "font": {
        "size": 16
        }
    },
    "edges": {
        "color": {
        "inherit": true
        },
        "smooth": false
    },
    "physics": {
        "barnesHut": {
        "gravitationalConstant": -80000,
        "springLength": 250
        },
        "minVelocity": 0.75
    }
    }
    """)

    # --- Generar y guardar el grafo en un archivo HTML ---
    net.write_html(f"./grafos_recomendacion/Flow_recomendation_{chat_id}.html")

    driver.close()


    # Abrir el archivo HTML en el navegador
    # Asegúrate de que la ruta sea absoluta
    ruta_absoluta = os.path.abspath(f"./grafos_recomendacion/Flow_recomendation_{chat_id}.html")

    # Abre el archivo en el navegador
    webbrowser.open(f'file://{ruta_absoluta}')

def verifify_subcadena(cadena_principal, subcadena,to_return):
    if subcadena in cadena_principal:
        return to_return
    else:
        return cadena_principal


def recomendacion(model:str, chat_id:str,info_poligono:dict,human_input='Genérame la recomendación') -> str:

    template = """
    Eres un experto técnico en infraestructura eléctrica. Tu objetivo es dar recomendaciones y pautas 
    normativas basadas en el contexto que se te proporciona. 

    Instrucciones para tu respuesta:
    1. Identifica las variables y sus valores que menciona el usuario.
    2. Revisa el contexto normativo y la información histórica (chat_history) para entender las normas 
    o límites aplicables.
    3. Compara cada valor de la variable con las normas del contexto, explicando si cumple o no cumple. 
    - Si no cumple, explica el riesgo o la consecuencia y brinda recomendaciones claras y accionables 
        (qué cambiar, qué revisar, qué reforzar, etc.). Además dar siempre una sugerencia de la normativa a la que se debe 
        ajustar y el valor al que se debe ajustar.
    - Si cumple, explica por qué cumple y qué se debe tener en cuenta a futuro (mantenimiento, 
        límites de uso, etc.).
    4. Dar siempre una sugerencia de la normativa a la que se debe ajustar la variable.
    5. Dar siempre explicitamente el valor al que se debe ajustar la variable o el rango de valores en el que debe estar.
    6. Redacta tu respuesta en un tono claro, pero conversacional y cercano, sin usar una lista demasiado 
    rígida. Estructura el texto en párrafos o viñetas libres para que sea más fácil de entender.
    7. Si existe ambigüedad o falta de información, explica qué información adicional sería necesaria 
    para dar una recomendación más completa.

    Usa el siguiente contexto y el historial de la conversación para redactar la recomendación:

    {context}

    {chat_history}

    Human: {human_input}

    Chatbot (RESPUESTA RECOMENDACIÓN UN SOLO PÁRRAFO):
    """

    
    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=template
    )
    

    if model=="gpt":
        try:
            llm_chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        except TypeError as e:
            if "Expected an instance of `httpx.Client` but got <class 'httpx.AsyncClient'>" in str(e):
                print("Error: Se esperaba una instancia de httpx.Client pero se proporcionó httpx.AsyncClient.")
                llm_chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
            else:
                raise e
    elif model=="llama1":
        llm_chat=ChatOllama(model="llama3.1",temperature=0)
    elif model=="llama2":
        llm_chat=ChatOllama(model="llama3.2:1b",temperature=0)


    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") #word2vec model of openAI
    responses=[]
    docs_all=[]
    for muestra in info_poligono.keys():
        memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
        chain = load_qa_chain(llm_chat, chain_type="stuff", memory=memory, prompt=prompt)
        # Load the chat history of the conversation for every particular agent
        path_memory=f"memories/{chat_id}.pkl"
        
        '''if os.path.exists(path_memory):
            with open(path_memory, 'rb') as f:
                memory = pickle.load(f) #memory of the conversation
            
            chain.memory=memory'''
        tipo_equipo=info_poligono[muestra]["Tipo_de_equipo"]
        variables_recomendacion=pd.read_excel(os.path.join("..", "data", "arbol_decision_recomendaciones", f"variables_{tipo_equipo}.xlsx"))
        docs=[]
        partes=[]
        for variable in info_poligono[muestra]["top_5"].keys():
            variable_original=variable
            if variable in list(variables_recomendacion["Variables"]):
                pass
            else:
                variable_modificada = verifify_subcadena(variable, 'pres','Presión Atmosférica')
                if variable_modificada == variable:
                    variable_modificada = verifify_subcadena(variable, 'rh','Humedad Relativa')
                    if variable_modificada == variable:
                        variable_modificada = verifify_subcadena(variable, 'slp','Presión a Nivel del Mar')
                        if variable_modificada == variable:
                            variable_modificada = verifify_subcadena(variable, 'solar_rad','Radiación Solar')
                            if variable_modificada == variable:
                                variable_modificada = verifify_subcadena(variable, 'temp','Temperatura Ambiente')
                                if variable_modificada == variable:
                                    variable_modificada = verifify_subcadena(variable, 'uv','Índice UV')
                                    if variable_modificada == variable:
                                        variable_modificada = verifify_subcadena(variable, 'vis','Visibilidad')
                                        if variable_modificada == variable:
                                            variable_modificada = verifify_subcadena(variable, 'wind_gust_spd','Ráfagas de Viento')
                                            if variable_modificada == variable:
                                                variable_modificada = verifify_subcadena(variable, 'wind_spd','Velocidad Promedio del Viento')
                                            else:
                                                variable = variable_modificada
                                        else:
                                            variable = variable_modificada
                                    else:
                                        variable = variable_modificada
                                else:
                                    variable = variable_modificada
                            else:
                                variable = variable_modificada
                        else:
                            variable = variable_modificada
                    else:
                        variable = variable_modificada
                else:
                    variable = variable_modificada
                
            if variable in ["ALTITUD_mean","ALTITUD_median","ALTITUD_min","ALTITUD_max","ALTITUD_std","CORRIENTE_mean","CORRIENTE_median","CORRIENTE_min","CORRIENTE_max","CORRIENTE_std","TIPO_1_count","TIPO_2_count"]:
                responses.append("\n")
            else:
                memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
                try:
                    documento_buscar=variables_recomendacion[variables_recomendacion["Variables"]==variable]["Documento "].iloc[0]
                except:
                    documento_buscar=variables_recomendacion[variables_recomendacion["Variables"]==variable]["Documento"].iloc[0]
                sugerencia=variables_recomendacion[variables_recomendacion["Variables"]==variable]["Sugerencia"].iloc[0]
                seccion_buscar=variables_recomendacion[variables_recomendacion["Variables"]==variable]["Normativa"].iloc[0]
                valor_variable=info_poligono[muestra]["top_5"][variable_original]
                
                # load from disk
                vectorstore = Chroma(persist_directory=f"./embeddings_by_procces/{documento_buscar}",embedding_function=embeddings)
                query=sugerencia+" "+seccion_buscar

                docs_variable=vectorstore.similarity_search(query,k=5) #Retriever
                docs=docs+docs_variable

                query=f"Generame una recomendación para la variable {variable}, la cual tiene un valor de {valor_variable}. {sugerencia}"

                response=chain({"input_documents": docs, "human_input": query, "chat_history":memory}, #,"sugerencia":sugerencia},
                        return_only_outputs=False) #AI answer

                response = response['output_text']

                responses.append(response)
        

        docs_all=docs_all+docs

        
        #Save the chat history (memory) for a new iteration of the conversation for the general agent:
        with open(path_memory, 'wb') as f:
            pickle.dump(chain.memory, f)


        with open("answer.pkl", 'wb') as archivo:
            pickle.dump(response, archivo)
    
    final_recommendation = "A continuación, se muestran las recomendaciones para cada equipo:\n\n"
    for idx, resp in enumerate(responses, start=0):
        if (idx==0 or idx==5 or idx==10):
          final_recommendation += "\n"
          final_recommendation += f"RECOMENDACIÓN PARA LAS VARIABLES DEL EQUIPO {info_poligono[list(info_poligono.keys())[int(idx/5)]]['Tipo_de_equipo'].upper()} CON ID {list(info_poligono.keys())[int(idx/5)]}: \n\n{resp}\n"
        else:
          final_recommendation+=f"\n{resp}\n"

    if variable in ["ALTITUD_mean","ALTITUD_median","ALTITUD_min","ALTITUD_max","ALTITUD_std","CORRIENTE_mean","CORRIENTE_median","CORRIENTE_min","CORRIENTE_max","CORRIENTE_std","TIPO_1_count","TIPO_2_count"]:
        final_recommendation+=f"\nDado que existe una posibilidad de que la interrupción esté provocada por una descarga eléctrica, se recomienda realizar inspecciones periódicas para detectar degradación en aisladores y equipos de protección, optimizar el sistema de puesta a tierra para disipar mejor la energía, y verificar la correcta coordinación de protecciones como reconectadores y pararrayos. Además, en zonas de alta incidencia de descargas, se sugiere reforzar el aislamiento y considerar la instalación de blindajes adicionales. La implementación de monitoreo en tiempo real y análisis de datos históricos facilitará la identificación de patrones y la planificación del mantenimiento preventivo. Finalmente, el uso de herramientas de predicción climática permitirá anticipar eventos críticos y tomar medidas proactivas para minimizar el impacto en la red de distribución.\n"


    if human_input == 'Genérame la recomendación':
        variables_por_equipo, normativas_por_equipo, documentos_por_equipo, sugerencias_por_equipo = get_structure_graph_recomendation(info_poligono)
        get_html_graph_recomendation(variables_por_equipo,normativas_por_equipo,documentos_por_equipo,sugerencias_por_equipo,chat_id)
        final_response=final_recommendation
    else:
        query=human_input
        final_response==chain({"input_documents": docs_all, "human_input": query, "chat_history":memory}, #,"sugerencia":sugerencia},
                    return_only_outputs=False) #AI answer
        
        
    return final_response


def recomendacion_apoyos(model:str, chat_id:str,data_equipo:dict,human_input='Genérame la recomendación') -> str:

    instruction="Basándote en el contexto normativo, los valores proporcionados y la sugerencia, por favor proporciona una recomendación detallada y justificada sobre la variable"
    variable_recomendacion=data_equipo["Variable_Recomendacion"]
    #variables_recomendacion_apoyos=pd.read_excel("C:/Users/spine/Downloads/variables_apoyos.xlsx")
    
    variables_recomendacion_apoyos=pd.read_excel(os.path.join("..", "data", "arbol_decision_recomendaciones", "variables_apoyos.xlsx"))
    sugerencia=variables_recomendacion_apoyos[variables_recomendacion_apoyos["Variables"]==variable_recomendacion]["Sugerencia"].iloc[0]
    seccion_buscar=variables_recomendacion_apoyos[variables_recomendacion_apoyos["Variables"]==variable_recomendacion]["Normativa"].iloc[0]
    documento_buscar=variables_recomendacion_apoyos[variables_recomendacion_apoyos["Variables"]==variable_recomendacion]["Documento "].iloc[0]
    valores_variable=json.dumps(data_equipo["Variable_Valores"])
  
    
    template="""Eres un experto técnico en infraestructura eléctrica. Tu función es dar recomendaciones y pautas 
                normativas basadas en el contexto que se te proporciona. 

                De acuerdo al valor de la variable que menciona el usuario en su pregunta, sigue estos pasos:
                1. Identifica la variable y el valor que proporciona el usuario.
                2. Consulta el contexto normativo proporcionado (por ejemplo, normas mínimas o rangos admitidos).
                3. Compara el valor dado con las normas del contexto.
                - Si el valor NO cumple con la norma, debes decirlo claramente, explicar por qué no cumple y recomendar la acción necesaria (por ejemplo, aumentar la longitud, reemplazar el apoyo, etc.).
                - Si el valor SÍ cumple con la norma, debes confirmarlo y, de ser necesario, brindar información adicional que refuerce la recomendación.
                4. Presenta la respuesta de forma clara y directa. Si se requiere, enumera pasos o consideraciones técnicas.
                
                
                Usa el contexto y el historial de la conversación para 
                responder a las preguntas del usuario:

            {context}

            {chat_history}
            Human: {human_input}
            Chatbot (RESPUESTA RECOMENDACIÓN):
            """


    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=template
    )


    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

    if model=="gpt":
        llm_chat=ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    elif model=="llama1":
        llm_chat=ChatOllama(model="llama3.1",temperature=0)
    elif model=="llama2":
        llm_chat=ChatOllama(model="llama3.2:1b",temperature=0)

    
    chain = load_qa_chain(llm_chat, chain_type="stuff", memory=memory, prompt=prompt)

    # Load the chat history of the conversation for every particular agent
    path_memory=f"memories/{chat_id}.pkl"
    
    
    if os.path.exists(path_memory):
        with open(path_memory, 'rb') as f:
            memory = pickle.load(f) #memory of the conversation
        
        chain.memory=memory

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") #word2vec model of openAI
    # load from disk
    
    vectorstore = Chroma(persist_directory=f"./embeddings_by_procces/{documento_buscar}",embedding_function=embeddings)
    
    if human_input == 'Génerame la recomendación':
        query =sugerencia+" "+seccion_buscar
    else: 
        query = human_input

    docs=vectorstore.similarity_search(query,k=5) #Retriever
    print(docs)
    
    if human_input == 'Genérame la recomendación':
        query=f"Generame una recomendación para la variable {variable_recomendacion}, la cual tiene un valor de {data_equipo['Variable_Valores'][variable_recomendacion]}. {sugerencia}"
    else:
        query=human_input

    response_all=chain({"input_documents": docs, "human_input": query, "chat_history":memory}, #,"sugerencia":sugerencia},
                    return_only_outputs=False) #AI answer
    

    response=response_all['output_text']

    
    #Save the chat history (memory) for a new iteration of the conversation for the general agent:
    with open(path_memory, 'wb') as f:
        pickle.dump(chain.memory, f)


    with open("answer.pkl", 'wb') as archivo:
        pickle.dump(response, archivo)

    return response

def recomendacion_switches(model:str, chat_id:str,data_equipo:dict,human_input='Genérame la recomendación') -> str:

    instruction="Basándote en el contexto normativo, los valores proporcionados y la sugerencia, por favor proporciona una recomendación detallada y justificada sobre la variable"
    variable_recomendacion=data_equipo["Variable_Recomendacion"]
    
    variables_recomendacion_apoyos=pd.read_excel(os.path.join("..", "data", "arbol_decision_recomendaciones", "variables_switches.xlsx"))
    sugerencia=variables_recomendacion_apoyos[variables_recomendacion_apoyos["Variables"]==variable_recomendacion]["Sugerencia"].iloc[0]
    seccion_buscar=variables_recomendacion_apoyos[variables_recomendacion_apoyos["Variables"]==variable_recomendacion]["Seccion_switches"].iloc[0]
    valores_variable=json.dumps(data_equipo["Variable_Valores"])

    template = """
    Contexto Normativo y Situación:
    {context}

    Historial de Conversación:
    {chat_history}

    Datos del Switche:
    {valores_variables}

    Situación Reportada:
    En el switche especìficao especificado, se produjo una interrupción que afectó la continuidad del servicio eléctrico. 
    Es fundamental garantizar que esta situación no se repita en el futuro. Las recomendaciones deben considerar 
    tanto el cumplimiento normativo como la prevención de interrupciones similares.

    Variable para la Recomendación:
    {variable_recomendacion}

    Sugerencia para la Recomendación:
    {sugerencia}

    Pregunta Actual del Usuario:
    {human_input}

    Tarea:
    1. Analiza el contexto normativo y determina los requisitos específicos que afectan a la variable "{variable_recomendacion}".
    2. Evalúa los valores proporcionados en "Datos del Switche" en relación con la interrupción reportada y 
       determina si cumplen con las normativas y buenas prácticas para prevenir interrupciones.
    3. Proporciona ejemplos concretos, indicando:
        - Valores óptimos según las normativas para la variable "{variable_recomendacion}".
        - Ejemplos de valores que podrían generar riesgo de interrupciones futuras.
        - Alternativas o modificaciones recomendadas para evitar interrupciones similares.
    4. Responde a la pregunta del usuario ({human_input}) considerando la información disponible.
    5. Explica claramente la lógica detrás de la recomendación, utilizando fragmentos del contexto normativo para respaldarla.
    6. Si es necesario, incluye acciones de mantenimiento, refuerzo estructural u otras estrategias para mitigar riesgos.

    Resultado esperado:
    Una recomendación detallada y justificada sobre la variable "{variable_recomendacion}", enfocada en prevenir interrupciones futuras, 
    con ejemplos específicos, sugerencias prácticas, y una respuesta clara a la pregunta planteada por el usuario.
    """


    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "valores_variables","variables_recomendacion","sugerencia"], template=template
    )


    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

    if model=="gpt":
        llm_chat=ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    elif model=="llama1":
        llm_chat=ChatOllama(model="llama3.1",temperature=0)
    elif model=="llama2":
        llm_chat=ChatOllama(model="llama3.2:1b",temperature=0)

    
    chain = load_qa_chain(llm_chat, chain_type="stuff", memory=memory, prompt=prompt)
    
    # Load the chat history of the conversation for every particular agent
    path_memory=f"memories/{chat_id}.pkl"
    if os.path.exists(path_memory):
        with open(path_memory, 'rb') as f:
            memory = pickle.load(f) #memory of the conversation
        
        chain.memory=memory

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") #word2vec model of openAI
    # load from disk
    vectorstore = Chroma(persist_directory=f"./embeddings_by_procces/normativa_switches",embedding_function=embeddings)
    
    if human_input == 'Génerame la recomendación':
        query =sugerencia+" "+seccion_buscar
    else: 
        query = human_input

    docs=vectorstore.similarity_search(query,k=5) #Retriever
    print(docs)

    response=chain({"input_documents": docs, "human_input": human_input, "chat_history":memory,
                    "valores_variables":valores_variable,"variable_recomendacion":variable_recomendacion,
                    "sugerencia":sugerencia},
                    return_only_outputs=False)['output_text'] #AI answer

    
    #Save the chat history (memory) for a new iteration of the conversation for the general agent:
    with open(path_memory, 'wb') as f:
        pickle.dump(chain.memory, f)


    with open("answer.pkl", 'wb') as archivo:
        pickle.dump(response, archivo)

    return response

def recomendacion_tramo_red(model:str, chat_id:str,data_equipo:dict,human_input='Genérame la recomendación') -> str:

    instruction="Basándote en el contexto normativo, los valores proporcionados y la sugerencia, por favor proporciona una recomendación detallada y justificada sobre la variable"
    variable_recomendacion=data_equipo["Variable_Recomendacion"]
    variables_recomendacion_apoyos=pd.read_excel(os.path.join("..", "data", "arbol_decision_recomendaciones", "variables_tramo_red.xlsx"))
    sugerencia=variables_recomendacion_apoyos[variables_recomendacion_apoyos["Variables"]==variable_recomendacion]["Sugerencia"].iloc[0]
    seccion_buscar=variables_recomendacion_apoyos[variables_recomendacion_apoyos["Variables"]==variable_recomendacion]["Seccion_tramo_red"].iloc[0]
    valores_variable=json.dumps(data_equipo["Variable_Valores"])

    template = """
    Contexto Normativo y Situación:
    {context}

    Historial de Conversación:
    {chat_history}

    Datos del Tramo de Red:
    {valores_variables}

    Situación Reportada:
    En el tramo de red especìficao especificado, se produjo una interrupción que afectó la continuidad del servicio eléctrico. 
    Es fundamental garantizar que esta situación no se repita en el futuro. Las recomendaciones deben considerar 
    tanto el cumplimiento normativo como la prevención de interrupciones similares.

    Variable para la Recomendación:
    {variable_recomendacion}

    Sugerencia para la Recomendación:
    {sugerencia}

    Pregunta Actual del Usuario:
    {human_input}

    Tarea:
    1. Analiza el contexto normativo y determina los requisitos específicos que afectan a la variable "{variable_recomendacion}".
    2. Evalúa los valores proporcionados en "Datos del Tramo de Red" en relación con la interrupción reportada y 
       determina si cumplen con las normativas y buenas prácticas para prevenir interrupciones.
    3. Proporciona ejemplos concretos, indicando:
        - Valores óptimos según las normativas para la variable "{variable_recomendacion}".
        - Ejemplos de valores que podrían generar riesgo de interrupciones futuras.
        - Alternativas o modificaciones recomendadas para evitar interrupciones similares.
    4. Responde a la pregunta del usuario ({human_input}) considerando la información disponible.
    5. Explica claramente la lógica detrás de la recomendación, utilizando fragmentos del contexto normativo para respaldarla.
    6. Si es necesario, incluye acciones de mantenimiento, refuerzo estructural u otras estrategias para mitigar riesgos.

    Resultado esperado:
    Una recomendación detallada y justificada sobre la variable "{variable_recomendacion}", enfocada en prevenir interrupciones futuras, 
    con ejemplos específicos, sugerencias prácticas, y una respuesta clara a la pregunta planteada por el usuario.
    """


    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "valores_variables","variables_recomendacion","sugerencia"], template=template
    )


    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

    if model=="gpt":
        llm_chat=ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    elif model=="llama1":
        llm_chat=ChatOllama(model="llama3.1",temperature=0)
    elif model=="llama2":
        llm_chat=ChatOllama(model="llama3.2:1b",temperature=0)

    
    chain = load_qa_chain(llm_chat, chain_type="stuff", memory=memory, prompt=prompt)
    
    # Load the chat history of the conversation for every particular agent
    path_memory=f"memories/{chat_id}.pkl"
    if os.path.exists(path_memory):
        with open(path_memory, 'rb') as f:
            memory = pickle.load(f) #memory of the conversation
        
        chain.memory=memory

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") #word2vec model of openAI
    # load from disk
    vectorstore = Chroma(persist_directory=f"./embeddings_by_procces/normativa_tramo_red",embedding_function=embeddings)
    
    if human_input == 'Génerame la recomendación':
        query =sugerencia+" "+seccion_buscar
    else: 
        query = human_input

    docs=vectorstore.similarity_search(query,k=5) #Retriever
    print(docs)

    response=chain({"input_documents": docs, "human_input": human_input, "chat_history":memory,
                    "valores_variables":valores_variable,"variable_recomendacion":variable_recomendacion,
                    "sugerencia":sugerencia},
                    return_only_outputs=False)['output_text'] #AI answer

    
    #Save the chat history (memory) for a new iteration of the conversation for the general agent:
    with open(path_memory, 'wb') as f:
        pickle.dump(chain.memory, f)


    with open("answer.pkl", 'wb') as archivo:
        pickle.dump(response, archivo)

    return response

def obtener_clave_maximo_score(diccionario):
    """
    Retorna la clave con el mayor valor en el diccionario.
    
    :param diccionario: dict, donde los valores son numéricos.
    :return: clave con el mayor valor.
    """
    if not diccionario:
        return None  # Retorna None si el diccionario está vacío
    
    clave_maxima = max(diccionario, key=diccionario.get)
    return clave_maxima

def recomendacion_transformadores(model:str, chat_id:str,data_equipo:dict,human_input='Genérame la recomendación') -> str:

    instruction="Basándote en el contexto normativo, los valores proporcionados y la sugerencia, por favor proporciona una recomendación detallada y justificada sobre la variable"
    variable_recomendacion=data_equipo["Variable_Recomendacion"]
    variables_recomendacion_apoyos=pd.read_excel(os.path.join("..", "data", "arbol_decision_recomendaciones", "variables_transformadores.xlsx"))
    sugerencia=variables_recomendacion_apoyos[variables_recomendacion_apoyos["Variables"]==variable_recomendacion]["Sugerencia"].iloc[0]
    seccion_buscar=variables_recomendacion_apoyos[variables_recomendacion_apoyos["Variables"]==variable_recomendacion]["Seccion_transformadores"].iloc[0]
    valores_variable=json.dumps(data_equipo["Variable_Valores"])

    template = """
    Contexto Normativo y Situación:
    {context}

    Historial de Conversación:
    {chat_history}

    Datos del Transformador:
    {valores_variables}

    Situación Reportada:
    En el transformador especificado, se produjo una interrupción que afectó la continuidad del servicio eléctrico. 
    Es fundamental garantizar que esta situación no se repita en el futuro. Las recomendaciones deben considerar 
    tanto el cumplimiento normativo como la prevención de interrupciones similares.

    Variable para la Recomendación:
    {variable_recomendacion}

    Sugerencia para la Recomendación:
    {sugerencia}

    Pregunta Actual del Usuario:
    {human_input}

    Tarea:
    1. Analiza el contexto normativo y determina los requisitos específicos que afectan a la variable "{variable_recomendacion}".
    2. Evalúa los valores proporcionados en "Datos del Transformador" en relación con la interrupción reportada y 
       determina si cumplen con las normativas y buenas prácticas para prevenir interrupciones.
    3. Proporciona ejemplos concretos, indicando:
        - Valores óptimos según las normativas para la variable "{variable_recomendacion}".
        - Ejemplos de valores que podrían generar riesgo de interrupciones futuras.
        - Alternativas o modificaciones recomendadas para evitar interrupciones similares.
    4. Responde a la pregunta del usuario ({human_input}) considerando la información disponible.
    5. Explica claramente la lógica detrás de la recomendación, utilizando fragmentos del contexto normativo para respaldarla.
    6. Si es necesario, incluye acciones de mantenimiento, refuerzo estructural u otras estrategias para mitigar riesgos.

    Resultado esperado:
    Una recomendación detallada y justificada sobre la variable "{variable_recomendacion}", enfocada en prevenir interrupciones futuras, 
    con ejemplos específicos, sugerencias prácticas, y una respuesta clara a la pregunta planteada por el usuario.
    """


    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "valores_variables","variables_recomendacion","sugerencia"], template=template
    )


    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

    if model=="gpt":
        llm_chat=ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    elif model=="llama1":
        llm_chat=ChatOllama(model="llama3.1",temperature=0)
    elif model=="llama2":
        llm_chat=ChatOllama(model="llama3.2:1b",temperature=0)

    
    chain = load_qa_chain(llm_chat, chain_type="stuff", memory=memory, prompt=prompt)
    
    # Load the chat history of the conversation for every particular agent
    path_memory=f"memories/{chat_id}.pkl"
    if os.path.exists(path_memory):
        with open(path_memory, 'rb') as f:
            memory = pickle.load(f) #memory of the conversation
        
        chain.memory=memory

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") #word2vec model of openAI
    # load from disk
    vectorstore = Chroma(persist_directory=f"./embeddings_by_procces/normativa_transformadores",embedding_function=embeddings)
    
    if human_input == 'Génerame la recomendación':
        query =sugerencia+" "+seccion_buscar
    else: 
        query = human_input

    docs=vectorstore.similarity_search(query,k=5) #Retriever
    print(docs)

    response=chain({"input_documents": docs, "human_input": human_input, "chat_history":memory,
                    "valores_variables":valores_variable,"variable_recomendacion":variable_recomendacion,
                    "sugerencia":sugerencia},
                    return_only_outputs=False)['output_text'] #AI answer

    
    #Save the chat history (memory) for a new iteration of the conversation for the general agent:
    with open(path_memory, 'wb') as f:
        pickle.dump(chain.memory, f)


    with open("answer.pkl", 'wb') as archivo:
        pickle.dump(response, archivo)

    return response

def get_recommendations(data_equipo):

    

    # 1. Leer datos una sola vez al inicio
    if os.path.exists(CHAT_DATA_FILE):
        with open(CHAT_DATA_FILE, 'r', encoding='utf-8') as archivo:
            try:
                data = json.load(archivo)
            except json.JSONDecodeError:
                print("Error: El archivo JSON está corrupto o vacío. Inicializando datos.")
                data = {"chats": {}, "current_chat_id": None}
    else:
        data = {"chats": {}, "current_chat_id": None}

    # 2. Inicializar nuevo chat
    new_chat_id = f'chat-{len(data["chats"])}'
    data['chats'][new_chat_id] = {'nombre': new_chat_id, 'mensajes': [], 'files': []}
    data['current_chat_id'] = new_chat_id

    # 3. Guardar datos del equipo
    os.makedirs('./body_recommendations', exist_ok=True)
    with open(f'./body_recommendations/{new_chat_id}.json', 'w') as json_file:
        json.dump(data_equipo, json_file)

    # 4. Agregar mensaje del usuario
    mensaje_user = {
        'autor': 'Tú',
        'texto': "",
        'needs_response': True,
        'modelo': "gpt",
        'proceso': "recomendacion"
    }
    data['chats'][new_chat_id]['mensajes'].append(mensaje_user)

    # 5. Guardar estado intermedio
    save_conversations(data)  # Guarda el estado antes de agregar la respuesta

    # 6. Agregar respuesta del asistente
    respuesta_asistente = {
        'autor': 'Asistente',
        'texto': recomendacion('gpt', new_chat_id, data_equipo)
    }
    data['chats'][new_chat_id]['mensajes'].append(respuesta_asistente)
    data['chats'][new_chat_id]['mensajes'][-2]['needs_response'] = False

    # 7. Verificar antes de guardar
    print("Datos a guardar en el archivo JSON:")
    print(json.dumps(data, ensure_ascii=False, indent=4))

    # 8. Guardar estado final
    save_conversations(data)
    numero = 1 
    with open('./options/count_chat.json', 'w') as archivo_json:
        json.dump(numero, archivo_json)


    
        