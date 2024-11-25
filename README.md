# Dashboard_CHEC


## Instrucciones para instalar el Chatbot

Sigue los siguientes pasos para instalar y ejecutar el dashboard:

1. **Descargar Ollama:** Para levantar un servidor que permita correr los LLMs de Meta, en este casos LLama 3.1 de 8b de parámetros y Llama 3.2 de 1b de parámetros. Esta descarga se puedes hacer desde: https://ollama.com/download, dependiendo de si la instalación se hará en una máquina con sistema operativo Windows, Linux o MacOs.

2. **Instalación de Ollama:** Una vez descargado el archivo .exe, proceder a instalar Ollama en la máquina.

3. **Descargar los Pesos de los Modelos de META:** Para descargar los pesos de los modelos de META mencionados anteriormente, se debe abrir una terminal de linea de comandos y correr los siguientes comandos:

   ```bash
   ollama pull llama3.1:8b
   ```

   ```bash
   ollama pull llama3.2:1b
   ```

4. **Levantar Servidor de Ollama:** Se procede a levantar el servidor de Ollama que permitirá consumir los modelos mencionados anteriormente através de una API, para esto se debe abrir una terminal de linea de comando y correr el siguiente comando:

   ```bash
   ollama serve
   ```

Posterior a esto, se recomienda reiniciar la máquina.

5. **Clonar el repositorio**

   ```bash
   git clone https://github.com/UN-GCPDS/chatbot-chec.git
   ```

2. **Ir al directorio del repositorio**

   ```bash
   cd chatbot-chec
   ```

3. **Crear un ambiente virtual**

   ```bash
   python -m venv venv
   ```

4. **Activar el ambiente virtual**

   - **En Windows:**

     ```bash
     venv\Scripts\activate
     ```

   - **En macOS/Linux:**

     ```bash
     source venv/bin/activate
     ```

5. **Instalar los requerimientos**

   ```bash
   pip install -r requirements.txt
   ```

6. **Ejecutar la aplicación**

   ```bash
   python main.py
   ```
