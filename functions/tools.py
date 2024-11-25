#PACKAGES:
from langchain.tools import tool
import json
import os
import pickle
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import OpenAI
import pandas as pd


@tool
def capitulo_1(query: str, model:str, chat_id:str) -> str:
    """
    Usar cuando se necesite responder preguntas acerca del capítulo 1 del RETIE.
    El capítulo 1 del RETIE establece las medidas para garantizar la seguridad de las personas, 
    la vida animal y vegetal, y la preservación del medio ambiente en relación con los riesgos de origen 
    eléctrico. Además, se asegura de que los sistemas, instalaciones, equipos y productos utilizados en la
    generación, transmisión, transformación, distribución y uso final de la energía eléctrica cumplan 
    con objetivos legítimos como la protección de la vida y la salud humana, animal y vegetal, la 
    prevención de prácticas que puedan inducir a error al usuario, entre otros. 
    También se establecen responsabilidades para diseñadores, constructores, operadores, propietarios, 
    fabricantes, importadores y distribuidores de materiales eléctricos, así como entidades encargadas 
    de la evaluación de la conformidad.
    """

    # Abrir el archivo en modo de lectura
    with open(f"number_iteration.pkl", "r") as archivo:
        # Leer el contenido del archivo
        contenido = archivo.read()
        # Convertir el contenido a un número entero
        number_iteration = int(contenido)
        
    # Abrir el archivo en modo de escritura
    with open(f"number_iteration.pkl", "w") as archivo:
        # Escribir el número entero en el archivo
        number_iteration=number_iteration+1
        archivo.write(str(number_iteration))
        
    #PROMP TEMPLATE:
    template = """ Se te proporcionará una serie de textos que contienen instrucciones sobre cómo 
                resolver preguntas acerca de normativas en redes eléctricas de nivel de tensión 2. Según estos textos,
                responde a la pregunta de la manera más completa posible.

                Dado el siguiente contexto y teniendo en cuenta el historial de la conversación, 
                responde a las preguntas hechas por el usuario:

                {context}

                {chat_history}
                Human: {human_input}
                Chatbot (RESPUESTA FORMAL):
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
    vectorstore = Chroma(persist_directory=f"embeddings_by_procces/capitulo_1",embedding_function=embeddings)

    docs=vectorstore.similarity_search(query,k=5) #Retriever

    print(docs)

    response=chain({"input_documents": docs, "human_input": query, "chat_history":memory}, return_only_outputs=False)['output_text'] #AI answer

    #Save the chat history (memory) for a new iteration of the conversation for the general agent:
    with open(path_memory, 'wb') as f:
        pickle.dump(chain.memory, f)

    
    # Abre el archivo en modo binario de escritura y guarda la cadena
    with open("answer.pkl", 'wb') as archivo:
        pickle.dump(response, archivo)

    return response


@tool
def capitulo_2(query: str, model:str, chat_id:str) -> str:
    """
    Usar cuando se necesite responder preguntas acerca del capítulo 2 del RETIE.
    El capítulo 2 del RETIE establece los requisitos y ensayos mínimos aplicables a los equipos y 
    productos utilizados en instalaciones eléctricas, con el fin de promover su adecuada utilización 
    fijando los parámetros mínimos de calidad, desempeño y seguridad. 
    Este capítulo también garantiza la protección de la vida y la salud humana, 
    la protección del medio ambiente, la prevención de prácticas que puedan inducir a error al usuario,
    y el uso racional y eficiente de la energía. 
    Además, se basa en objetivos específicos como unificar los requisitos de seguridad para los
    productos eléctricos de mayor utilización, prevenir actos que induzcan a error a los usuarios, y 
    exigir requisitos para contribuir al uso racional y eficiente de la energía. También se establecen 
    normas para la certificación de productos, marcaciones, rotulados, y responsabilidades de los 
    productores y comercializadores.
    """


    # Abrir el archivo en modo de lectura
    with open(f"number_iteration.pkl", "r") as archivo:
        # Leer el contenido del archivo
        contenido = archivo.read()
        # Convertir el contenido a un número entero
        number_iteration = int(contenido)
        
    # Abrir el archivo en modo de escritura
    with open(f"number_iteration.pkl", "w") as archivo:
        # Escribir el número entero en el archivo
        number_iteration=number_iteration+1
        archivo.write(str(number_iteration))

    #PROMP TEMPLATE:
    template = """ Se te proporcionará una serie de textos que contienen instrucciones sobre cómo 
                resolver preguntas acerca de normativas en redes eléctricas de nivel de tensión 2. Según estos textos,
                responde a la pregunta de la manera más completa posible.

                Dado el siguiente contexto y teniendo en cuenta el historial de la conversación, 
                responde a las preguntas hechas por el usuario:

                {context}

                {chat_history}
                Human: {human_input}
                Chatbot (RESPUESTA FORMAL):
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
    vectorstore = Chroma(persist_directory=f"embeddings_by_procces/capitulo_2",embedding_function=embeddings)

    docs=vectorstore.similarity_search(query,k=5) #Retriever

    print(docs)

    response=chain({"input_documents": docs, "human_input": query, "chat_history":memory}, return_only_outputs=False)['output_text'] #AI answer

    #Save the chat history (memory) for a new iteration of the conversation for the general agent:
    with open(path_memory, 'wb') as f:
        pickle.dump(chain.memory, f)

    with open("answer.pkl", 'wb') as archivo:
        pickle.dump(response, archivo)

    return response


@tool
def capitulo_3(query: str, model:str, chat_id:str) -> str:
    """
    Usar cuando se necesite responder preguntas acerca del capítulo 3 del RETIE.
    El Capítulo 3 del RETIE se refiere a los requisitos generales de las instalaciones eléctricas. 
    En este capítulo se establecen normativas y criterios para el diseño, operación y mantenimiento de
    las instalaciones eléctricas, así como para la protección contra riesgos eléctricos. 
    Se abordan temas como las competencias y responsabilidades de las personas que intervienen 
    en las instalaciones eléctricas, el diseño de las instalaciones, los espacios para montaje de equipos,
    el código de colores para conductores, 
    entre otros aspectos relevantes para garantizar la seguridad y eficiencia de las redes eléctricas 
    de nivel de tensión 2.
    """

    # Abrir el archivo en modo de lectura
    with open(f"number_iteration.pkl", "r") as archivo:
        # Leer el contenido del archivo
        contenido = archivo.read()
        # Convertir el contenido a un número entero
        number_iteration = int(contenido)
        
    # Abrir el archivo en modo de escritura
    with open(f"number_iteration.pkl", "w") as archivo:
        # Escribir el número entero en el archivo
        number_iteration=number_iteration+1
        archivo.write(str(number_iteration))



    #PROMP TEMPLATE:
    template = """ Se te proporcionará una serie de textos que contienen instrucciones sobre cómo 
                resolver preguntas acerca de normativas en redes eléctricas de nivel de tensión 2. Según estos textos,
                responde a la pregunta de la manera más completa posible.

                Dado el siguiente contexto y teniendo en cuenta el historial de la conversación, 
                responde a las preguntas hechas por el usuario:

                {context}

                {chat_history}
                Human: {human_input}
                Chatbot (RESPUESTA FORMAL):
                """ 
    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=template
    )

    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
    print("MODEL")
    print(model)
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
    vectorstore = Chroma(persist_directory=f"embeddings_by_procces/capitulo_3",embedding_function=embeddings)

    docs=vectorstore.similarity_search(query,k=5) #Retriever

    print(docs)

    response=chain({"input_documents": docs, "human_input": query, "chat_history":memory}, return_only_outputs=False)['output_text'] #AI answer

    #Save the chat history (memory) for a new iteration of the conversation for the general agent:
    with open(path_memory, 'wb') as f:
        pickle.dump(chain.memory, f)

    
    with open("answer.pkl", 'wb') as archivo:
        pickle.dump(response, archivo)

    return response

@tool
def capitulo_4(query: str, model:str, chat_id:str) -> str:
    """
    Usar cuando se necesite responder preguntas acerca del capítulo 4 del RETIE.
    El capítulo 4 del RETIE aborda la Evaluación de la Conformidad en el Reglamento Técnico de 
    Instalaciones Eléctricas. Define los requisitos mínimos para los Certificados de Producto, 
    tales como la identificación clara de que es un certificado, el nombre del organismo certificador, 
    el esquema de certificación empleado, el número o referencia del certificado, así como 
    la identificación del productor, fabricante y del producto. También describe el alcance de la 
    certificación. Además, establece que el Ministerio de Minas y Energía de Colombia es la entidad 
    responsable de crear, revisar, actualizar e interpretar el RETIE, y detalla las sanciones para 
    quienes incumplan los requisitos del reglamento, incluyendo empresas de servicios públicos, 
    responsables de instalaciones eléctricas, usuarios, productores y laboratorios de pruebas.
    """


    # Abrir el archivo en modo de lectura
    with open(f"number_iteration.pkl", "r") as archivo:
        # Leer el contenido del archivo
        contenido = archivo.read()
        # Convertir el contenido a un número entero
        number_iteration = int(contenido)
        
    # Abrir el archivo en modo de escritura
    with open(f"number_iteration.pkl", "w") as archivo:
        # Escribir el número entero en el archivo
        number_iteration=number_iteration+1
        archivo.write(str(number_iteration))


    #PROMP TEMPLATE:
    template = """ Se te proporcionará una serie de textos que contienen instrucciones sobre cómo 
                resolver preguntas acerca de normativas en redes eléctricas de nivel de tensión 2. Según estos textos,
                responde a la pregunta de la manera más completa posible.

                Dado el siguiente contexto y teniendo en cuenta el historial de la conversación, 
                responde a las preguntas hechas por el usuario:

                {context}

                {chat_history}
                Human: {human_input}
                Chatbot (RESPUESTA FORMAL):
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
    vectorstore = Chroma(persist_directory=f"embeddings_by_procces/capitulo_4",embedding_function=embeddings)

    docs=vectorstore.similarity_search(query,k=5) #Retriever

    print(docs)

    response=chain({"input_documents": docs, "human_input": query, "chat_history":memory}, return_only_outputs=False)['output_text'] #AI answer

    #Save the chat history (memory) for a new iteration of the conversation for the general agent:
    with open(path_memory, 'wb') as f:
        pickle.dump(chain.memory, f)


    with open("answer.pkl", 'wb') as archivo:
        pickle.dump(response, archivo)

    return response


@tool
def resolucion_40117(query: str, model:str, chat_id:str) -> str:
    """
    Usar cuando se necesite responder preguntas acerca de la resolución resolucion 40117 del 02 de Abril de 2024.
    """

    # Abrir el archivo en modo de lectura
    with open(f"number_iteration.pkl", "r") as archivo:
        # Leer el contenido del archivo
        contenido = archivo.read()
        # Convertir el contenido a un número entero
        number_iteration = int(contenido)
        
    # Abrir el archivo en modo de escritura
    with open(f"number_iteration.pkl", "w") as archivo:
        # Escribir el número entero en el archivo
        number_iteration=number_iteration+1
        archivo.write(str(number_iteration))


    #PROMP TEMPLATE:
    template = """ Se te proporcionará una serie de textos que contienen instrucciones sobre cómo 
                resolver preguntas acerca de normativas en redes eléctricas de nivel de tensión 2. Según estos textos,
                responde a la pregunta de la manera más completa posible.

                Dado el siguiente contexto y teniendo en cuenta el historial de la conversación, 
                responde a las preguntas hechas por el usuario:

                {context}

                {chat_history}
                Human: {human_input}
                Chatbot (RESPUESTA FORMAL):
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
    vectorstore = Chroma(persist_directory=f"embeddings_by_procces/resolucion_40117",embedding_function=embeddings)

    docs=vectorstore.similarity_search(query,k=5) #Retriever

    print(docs)

    response=chain({"input_documents": docs, "human_input": query, "chat_history":memory}, return_only_outputs=False)['output_text'] #AI answer

    #Save the chat history (memory) for a new iteration of the conversation for the general agent:
    with open(path_memory, 'wb') as f:
        pickle.dump(chain.memory, f)


    with open("answer.pkl", 'wb') as archivo:
        pickle.dump(response, archivo)

    return response

@tool
def normativa_apoyos(query: str, model:str, chat_id:str) -> str:
    """
    Utilizar cuando se necesite responder preguntas acerca de los requisitos técnicos, normativos y 
    constructivos para el diseño, instalación y mantenimiento de apoyos y postes eléctricos en redes de
    distribución y transmisión, con énfasis en materiales, resistencia mecánica, protección anticorrosiva, 
    alturas, distancias de seguridad, y cumplimiento de normativas como el RETIE, NTC y ASTM, asegurando 
    estabilidad estructural y seguridad en diversas condiciones ambientales y operativas.
    """

    # Abrir el archivo en modo de lectura
    with open(f"number_iteration.pkl", "r") as archivo:
        # Leer el contenido del archivo
        contenido = archivo.read()
        # Convertir el contenido a un número entero
        number_iteration = int(contenido)
        
    # Abrir el archivo en modo de escritura
    with open(f"number_iteration.pkl", "w") as archivo:
        # Escribir el número entero en el archivo
        number_iteration=number_iteration+1
        archivo.write(str(number_iteration))


    #PROMP TEMPLATE:
    template = """ Se te proporcionará una serie de textos que contienen instrucciones sobre cómo 
                resolver preguntas acerca de normativas en redes eléctricas de nivel de tensión 2. 
                Según estos textos, responde a la pregunta de la manera más completa posible.

                Dado el siguiente contexto y teniendo en cuenta el historial de la conversación, 
                responde a las preguntas hechas por el usuario:

                {context}

                {chat_history}
                Human: {human_input}
                Chatbot (RESPUESTA FORMAL):
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
    vectorstore = Chroma(persist_directory=f"embeddings_by_procces/normativa_apoyos",embedding_function=embeddings)

    docs=vectorstore.similarity_search(query,k=5) #Retriever

    print(docs)

    response=chain({"input_documents": docs, "human_input": query, "chat_history":memory}, return_only_outputs=False)['output_text'] #AI answer

    #Save the chat history (memory) for a new iteration of the conversation for the general agent:
    with open(path_memory, 'wb') as f:
        pickle.dump(chain.memory, f)


    with open("answer.pkl", 'wb') as archivo:
        pickle.dump(response, archivo)

    return response

@tool
def normativa_protecciones(query: str, model:str, chat_id:str) -> str:
    """
    Utilizar cuando se necesite responder preguntas acerca de la selección, instalación y coordinación 
    de protecciones eléctricas mediante fusibles y dispositivos de protección contra sobretensiones (DPS), 
    considerando normativas específicas como la NTC 2797, NTC 2132, IEC 61643-1, UL 1449, y IEEE C62.41, 
    aplicables a transformadores, alimentadores y ramales. También es útil para determinar capacidades de 
    fusibles en sistemas de 13,2 kV y 34,5 kV, garantizar la protección contra sobrecorrientes y 
    sobretensiones, y asegurar la conformidad con estándares técnicos en sistemas eléctricos de 
    distribución.
    """

    # Abrir el archivo en modo de lectura
    with open(f"number_iteration.pkl", "r") as archivo:
        # Leer el contenido del archivo
        contenido = archivo.read()
        # Convertir el contenido a un número entero
        number_iteration = int(contenido)
        
    # Abrir el archivo en modo de escritura
    with open(f"number_iteration.pkl", "w") as archivo:
        # Escribir el número entero en el archivo
        number_iteration=number_iteration+1
        archivo.write(str(number_iteration))


    #PROMP TEMPLATE:
    template = """ Se te proporcionará una serie de textos que contienen instrucciones sobre cómo 
                resolver preguntas acerca de normativas en redes eléctricas de nivel de tensión 2. 
                Según estos textos, responde a la pregunta de la manera más completa posible.

                Dado el siguiente contexto y teniendo en cuenta el historial de la conversación, 
                responde a las preguntas hechas por el usuario:

                {context}

                {chat_history}
                Human: {human_input}
                Chatbot (RESPUESTA FORMAL):
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
    vectorstore = Chroma(persist_directory=f"embeddings_by_procces/normativa_protecciones",embedding_function=embeddings)

    docs=vectorstore.similarity_search(query,k=5) #Retriever

    print(docs)

    response=chain({"input_documents": docs, "human_input": query, "chat_history":memory}, return_only_outputs=False)['output_text'] #AI answer

    #Save the chat history (memory) for a new iteration of the conversation for the general agent:
    with open(path_memory, 'wb') as f:
        pickle.dump(chain.memory, f)


    with open("answer.pkl", 'wb') as archivo:
        pickle.dump(response, archivo)

    return response


@tool
def normativa_aisladores(query: str, model:str, chat_id:str) -> str:
    """
    Utilizar cuando se necesite responder preguntas acerca de la selección, uso y mantenimiento de 
    aisladores eléctricos en redes de distribución y transmisión, considerando normativas técnicas 
    (RETIE, IEC), parámetros de diseño (tensión, contaminación, esfuerzos mecánicos), tipos de aisladores 
    adecuados según el entorno, cálculos de distancia de fuga, y especificaciones técnicas para garantizar
    confiabilidad y seguridad en condiciones operativas y ambientales adversas.
    """

    # Abrir el archivo en modo de lectura
    with open(f"number_iteration.pkl", "r") as archivo:
        # Leer el contenido del archivo
        contenido = archivo.read()
        # Convertir el contenido a un número entero
        number_iteration = int(contenido)
        
    # Abrir el archivo en modo de escritura
    with open(f"number_iteration.pkl", "w") as archivo:
        # Escribir el número entero en el archivo
        number_iteration=number_iteration+1
        archivo.write(str(number_iteration))


    #PROMP TEMPLATE:
    template = """ Se te proporcionará una serie de textos que contienen instrucciones sobre cómo 
                resolver preguntas acerca de normativas en redes eléctricas de nivel de tensión 2. 
                Según estos textos, responde a la pregunta de la manera más completa posible.

                Dado el siguiente contexto y teniendo en cuenta el historial de la conversación, 
                responde a las preguntas hechas por el usuario:

                {context}

                {chat_history}
                Human: {human_input}
                Chatbot (RESPUESTA FORMAL):
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
    vectorstore = Chroma(persist_directory=f"embeddings_by_procces/normativa_aisladores",embedding_function=embeddings)

    docs=vectorstore.similarity_search(query,k=5) #Retriever

    print(docs)

    response=chain({"input_documents": docs, "human_input": query, "chat_history":memory}, return_only_outputs=False)['output_text'] #AI answer

    #Save the chat history (memory) for a new iteration of the conversation for the general agent:
    with open(path_memory, 'wb') as f:
        pickle.dump(chain.memory, f)


    with open("answer.pkl", 'wb') as archivo:
        pickle.dump(response, archivo)

    return response

@tool
def redes_aereas_media_tension(query: str, model:str, chat_id:str) -> str:
    """
    Utilizar cuando se necesite responder preguntas acerca de el diseño, construcción y mantenimiento de 
    redes aéreas de media tensión, abarcando criterios como tipos de líneas (33 kV y 13.2 kV), 
    especificaciones de apoyos primarios, distancias mínimas de seguridad, selección de conductores, 
    nivel de aislamiento, métodos de puesta a tierra, y normativas aplicables para garantizar la 
    seguridad, confiabilidad y conformidad con estándares técnicos en zonas urbanas y rurales.
    """

    # Abrir el archivo en modo de lectura
    with open(f"number_iteration.pkl", "r") as archivo:
        # Leer el contenido del archivo
        contenido = archivo.read()
        # Convertir el contenido a un número entero
        number_iteration = int(contenido)
        
    # Abrir el archivo en modo de escritura
    with open(f"number_iteration.pkl", "w") as archivo:
        # Escribir el número entero en el archivo
        number_iteration=number_iteration+1
        archivo.write(str(number_iteration))


    #PROMP TEMPLATE:
    template = """ Se te proporcionará una serie de textos que contienen instrucciones sobre cómo 
                resolver preguntas acerca de normativas en redes eléctricas de nivel de tensión 2. 
                Según estos textos, responde a la pregunta de la manera más completa posible.

                Dado el siguiente contexto y teniendo en cuenta el historial de la conversación, 
                responde a las preguntas hechas por el usuario:

                {context}

                {chat_history}
                Human: {human_input}
                Chatbot (RESPUESTA FORMAL):
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
    vectorstore = Chroma(persist_directory=f"embeddings_by_procces/redes_aereas_media_tension",embedding_function=embeddings)

    docs=vectorstore.similarity_search(query,k=5) #Retriever

    print(docs)

    response=chain({"input_documents": docs, "human_input": query, "chat_history":memory}, return_only_outputs=False)['output_text'] #AI answer

    #Save the chat history (memory) for a new iteration of the conversation for the general agent:
    with open(path_memory, 'wb') as f:
        pickle.dump(chain.memory, f)


    with open("answer.pkl", 'wb') as archivo:
        pickle.dump(response, archivo)

    return response

@tool
def codigo_electrico_colombiano(query: str, model:str, chat_id:str) -> str:
    """
    Utilizar cuando se necesite responder preguntas acerca de los requisitos técnicos, 
    normativos y de seguridad para instalaciones eléctricas en Colombia, incluyendo sistemas de 
    alambrado, protección contra sobrecorriente, puesta a tierra, acometidas, métodos de instalación, 
    selección de materiales, y cumplimiento del Código Eléctrico Colombiano (NTC 2050), aplicable a 
    proyectos residenciales, comerciales e industriales.
    """

    # Abrir el archivo en modo de lectura
    with open(f"number_iteration.pkl", "r") as archivo:
        # Leer el contenido del archivo
        contenido = archivo.read()
        # Convertir el contenido a un número entero
        number_iteration = int(contenido)
        
    # Abrir el archivo en modo de escritura
    with open(f"number_iteration.pkl", "w") as archivo:
        # Escribir el número entero en el archivo
        number_iteration=number_iteration+1
        archivo.write(str(number_iteration))


    #PROMP TEMPLATE:
    template = """ Se te proporcionará una serie de textos que contienen instrucciones sobre cómo 
                resolver preguntas acerca de normativas en redes eléctricas de nivel de tensión 2. 
                Según estos textos, responde a la pregunta de la manera más completa posible.

                Dado el siguiente contexto y teniendo en cuenta el historial de la conversación, 
                responde a las preguntas hechas por el usuario:

                {context}

                {chat_history}
                Human: {human_input}
                Chatbot (RESPUESTA FORMAL):
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
    vectorstore = Chroma(persist_directory=f"embeddings_by_procces/codigo_electrico_colombiano",embedding_function=embeddings)

    docs=vectorstore.similarity_search(query,k=5) #Retriever

    print(docs)

    response=chain({"input_documents": docs, "human_input": query, "chat_history":memory}, return_only_outputs=False)['output_text'] #AI answer

    #Save the chat history (memory) for a new iteration of the conversation for the general agent:
    with open(path_memory, 'wb') as f:
        pickle.dump(chain.memory, f)


    with open("answer.pkl", 'wb') as archivo:
        pickle.dump(response, archivo)

    return response


@tool
def requisitos_redes_aereas(query: str, model:str, chat_id:str) -> str:
    """
    Utilizar cuando se necesite responder preguntas acerca de los requisitos técnicos y normativos 
    para el diseño, construcción y mantenimiento de redes eléctricas aéreas en zonas especiales como 
    áreas costeras, contaminadas o con alta densidad de descargas atmosféricas, incluyendo la selección 
    de materiales (aisladores, postes, herrajes), métodos de construcción, 
    parámetros de aislamiento (CFO), y estrategias para mejorar la confiabilidad frente a condiciones 
    adversas, basándose en normativas como IEEE 1410, RETIE e IEC 60815.
    """

    # Abrir el archivo en modo de lectura
    with open(f"number_iteration.pkl", "r") as archivo:
        # Leer el contenido del archivo
        contenido = archivo.read()
        # Convertir el contenido a un número entero
        number_iteration = int(contenido)
        
    # Abrir el archivo en modo de escritura
    with open(f"number_iteration.pkl", "w") as archivo:
        # Escribir el número entero en el archivo
        number_iteration=number_iteration+1
        archivo.write(str(number_iteration))


    #PROMP TEMPLATE:
    template = """ Se te proporcionará una serie de textos que contienen instrucciones sobre cómo 
                resolver preguntas acerca de normativas en redes eléctricas de nivel de tensión 2. 
                Según estos textos, responde a la pregunta de la manera más completa posible.

                Dado el siguiente contexto y teniendo en cuenta el historial de la conversación, 
                responde a las preguntas hechas por el usuario:

                {context}

                {chat_history}
                Human: {human_input}
                Chatbot (RESPUESTA FORMAL):
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
    vectorstore = Chroma(persist_directory=f"embeddings_by_procces/requisitos_redes_aereas",embedding_function=embeddings)

    docs=vectorstore.similarity_search(query,k=5) #Retriever

    print(docs)

    response=chain({"input_documents": docs, "human_input": query, "chat_history":memory}, return_only_outputs=False)['output_text'] #AI answer

    #Save the chat history (memory) for a new iteration of the conversation for the general agent:
    with open(path_memory, 'wb') as f:
        pickle.dump(chain.memory, f)


    with open("answer.pkl", 'wb') as archivo:
        pickle.dump(response, archivo)

    return response

@tool
def retie(query: str, model:str, chat_id:str) -> str:
    """
    Utilizar cuando se necesite responder preguntas acerca de el diseño, instalación, operación, 
    y mantenimiento de instalaciones eléctricas en Colombia, según los requisitos del Reglamento 
    Técnico de Instalaciones Eléctricas (RETIE). Esto incluye temas como competencias de las personas 
    involucradas, diseño eléctrico detallado y básico, sistemas de puesta a tierra, protección contra 
    rayos, distancias de seguridad, clasificación de instalaciones, y cumplimiento de normativas de 
    seguridad eléctrica y ambiental.
    """

    # Abrir el archivo en modo de lectura
    with open(f"number_iteration.pkl", "r") as archivo:
        # Leer el contenido del archivo
        contenido = archivo.read()
        # Convertir el contenido a un número entero
        number_iteration = int(contenido)
        
    # Abrir el archivo en modo de escritura
    with open(f"number_iteration.pkl", "w") as archivo:
        # Escribir el número entero en el archivo
        number_iteration=number_iteration+1
        archivo.write(str(number_iteration))


    #PROMP TEMPLATE:
    template = """ Se te proporcionará una serie de textos que contienen instrucciones sobre cómo 
                resolver preguntas acerca de normativas en redes eléctricas de nivel de tensión 2. 
                Según estos textos, responde a la pregunta de la manera más completa posible.

                Dado el siguiente contexto y teniendo en cuenta el historial de la conversación, 
                responde a las preguntas hechas por el usuario:

                {context}

                {chat_history}
                Human: {human_input}
                Chatbot (RESPUESTA FORMAL):
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
    vectorstore = Chroma(persist_directory=f"embeddings_by_procces/retie",embedding_function=embeddings)

    docs=vectorstore.similarity_search(query,k=5) #Retriever

    print(docs)

    response=chain({"input_documents": docs, "human_input": query, "chat_history":memory}, return_only_outputs=False)['output_text'] #AI answer

    #Save the chat history (memory) for a new iteration of the conversation for the general agent:
    with open(path_memory, 'wb') as f:
        pickle.dump(chain.memory, f)


    with open("answer.pkl", 'wb') as archivo:
        pickle.dump(response, archivo)

    return response

@tool
def eventos_transformadores(query: str, model:str, chat_id:str) -> str:
    """
    Usar cuando se necesite responder preguntas acerca de eventos y/o interrupciones.
    """


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

    # Abrir el archivo en modo de lectura
    with open(f"number_iteration.pkl", "r") as archivo:
        # Leer el contenido del archivo
        contenido = archivo.read()
        # Convertir el contenido a un número entero
        number_iteration = int(contenido)
        
    # Abrir el archivo en modo de escritura
    with open(f"number_iteration.pkl", "w") as archivo:
        # Escribir el número entero en el archivo
        number_iteration=number_iteration+2
        archivo.write(str(number_iteration))
    
    # if model=="gpt":
    #     llm_agent=ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    # elif model=="llama1":
    #     llm_agent=ChatOllama(model="llama3.1",temperature=0)
    # elif model=="llama2":
    #     llm_agent=ChatOllama(model="llama3.2:1b",temperature=0)

    try:
        agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        eventos_trafos,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True,
        include_df_in_prompt=True,  # Incluye las primeras filas del DataFrame en el prompt
        number_of_head_rows=5)

        response=agent.invoke(query)["output"]

    except:
        response="De acuerdo a mi conocimiento actual, no tengo la capacidad para responder a tu pregunta, por favor reformula tu pregunta."


    with open("answer.pkl", 'wb') as archivo:
        pickle.dump(response, archivo)

    return response

@tool
def eventos_transformadores_plots(query: str, model:str, chat_id:str) -> str:
    """
    Usar cuando se necesite graficar acerca de eventos y/o interrupciones.
    """
    
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
    """
    # Al final, redacta conclusiones basadas **únicamente en los datos proporcionados en el DataFrame**.
    # Asegúrate de que todas las estadísticas y observaciones estén directamente derivadas de los datos.

    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), eventos_trafos, verbose=True, allow_dangerous_code=True)
   
    # Abrir el archivo en modo de lectura
    with open(f"number_iteration.pkl", "r") as archivo:
        # Leer el contenido del archivo
        contenido = archivo.read()
        # Convertir el contenido a un número entero
        number_iteration = int(contenido)

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
        # Abrir el archivo en modo de escritura
        with open(f"number_iteration.pkl", "w") as archivo:
            # Escribir el número entero en el archivo
            number_iteration=number_iteration+3
            archivo.write(str(number_iteration))

    except:
        response="De acuerdo a mi conocimiento actual, no tengo la capacidad para responder a tu pregunta, por favor reformula tu pregunta."
        # Abrir el archivo en modo de escritura
        with open(f"number_iteration.pkl", "w") as archivo:
            # Escribir el número entero en el archivo
            number_iteration=number_iteration+4
            archivo.write(str(number_iteration))

    
    
        
    

    with open("answer.pkl", 'wb') as archivo:
        pickle.dump(response, archivo)

    return response