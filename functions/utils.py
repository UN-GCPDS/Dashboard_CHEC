# functions/utils.py
import json
import os
import pickle
import pandas as pd
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
from langchain_community.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import OpenAI
import time
import shutil 

from functions.procces import eventos_transformadores_procces,eventos_transformadores_plots_procces
from functions.tools import capitulo_1, capitulo_2, capitulo_3, capitulo_4, resolucion_40117,eventos_transformadores, eventos_transformadores_plots, normativa_apoyos, normativa_protecciones, normativa_aisladores, redes_aereas_media_tension, codigo_electrico_colombiano, requisitos_redes_aereas, retie


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
                chain = load_qa_chain(ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0), chain_type="stuff", memory=memory, prompt=prompt)
            
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
    
        