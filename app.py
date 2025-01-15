import os
from dash import Dash

app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@700&display=swap',
        'https://fonts.googleapis.com/css2?family=Poppins:wght@700&display=swap'])

with open("C:/Users/lucas/OneDrive - Universidad Nacional de Colombia/PC-GCPDS/Documentos/data/OPENAI_API_Key.txt", "r") as archivo:
        OPENAI_API_KEY = archivo.read()

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

server = app.server