import os
from dash import Dash

app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@700&display=swap',
        'https://fonts.googleapis.com/css2?family=Poppins:wght@700&display=swap'])

os.environ['OPENAI_API_KEY'] = "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"

server = app.server