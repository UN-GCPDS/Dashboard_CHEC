from flask import send_from_directory
import dash
from dash import html

# Inicializa la aplicación Dash
app = dash.Dash(__name__)
server = app.server

# Ruta absoluta al directorio donde está el PDF
PDF_DIRECTORY = r"C:/Users/lucas/OneDrive - Universidad Nacional de Colombia/PC-GCPDS/Descargas"

# Configura la ruta estática para servir el archivo PDF
@server.route('/pdf/<path:filename>')
def serve_pdf(filename):
    return send_from_directory(PDF_DIRECTORY, filename)

# Diseña la interfaz de la aplicación Dash
app.layout = html.Div([
    html.H1("Visualizador de PDF en Dash"),
    html.Div(
        html.Iframe(
            src="/pdf/Lucas Iturriago - Hoja de vida.pdf",  # Ruta relativa al archivo servido
            style={"width": "50%", "height": "800px","position": "relative"},  # Ajusta las dimensiones
        ),
        style={"border": "1px solid #ccc", "margin": "20px 0", "display": "flex", "alignItems": "center", "justifyContent": "center"},
    )
])

# Ejecuta la aplicación
if __name__ == "__main__":
    app.run_server(debug=True)