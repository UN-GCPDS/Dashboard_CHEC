import math
import folium
import pandas as pd
import numpy as np
from datetime import datetime
from folium.plugins import HeatMap
import networkx as nx
from scipy.spatial import ConvexHull
from shapely.geometry import LineString, Point

def filtrar_aguas_abajo(grafo, nodo_inicial):
    aguas_abajo = set(nx.descendants(grafo, nodo_inicial))
    aguas_abajo.add(nodo_inicial)
    return aguas_abajo

def select_data(año,mes,mun,trafos,apoyos,switches,redmt,super_eventos, descargas, vegetacion):
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

    return trafos_seleccionado, apoyos_seleccionado, switches_seleccionado, redmt_seleccionado, super_eventos_seleccionado_1, descargas_seleccionado_1, vegetacion_seleccionado_1

def load_data():

    trafos = pd.read_pickle("C:/Users/lucas/OneDrive - Universidad Nacional de Colombia/PC-GCPDS/Documentos/data/TRAFOS_1.pkl")

    apoyos = pd.read_pickle("C:/Users/lucas/OneDrive - Universidad Nacional de Colombia/PC-GCPDS/Documentos/data/APOYOS_1.pkl")

    redmt = pd.read_pickle("C:/Users/lucas/OneDrive - Universidad Nacional de Colombia/PC-GCPDS/Documentos/data/REDMT_1.pkl")

    switches = pd.read_pickle("C:/Users/lucas/OneDrive - Universidad Nacional de Colombia/PC-GCPDS/Documentos/data/SWITCHES_1.pkl")

    super_eventos = pd.read_pickle("C:/Users/lucas/OneDrive - Universidad Nacional de Colombia/PC-GCPDS/Documentos/data/SuperEventos_Criticidad_AguasAbajo.pkl")

    descargas = pd.read_pickle("C:/Users/lucas/OneDrive - Universidad Nacional de Colombia/PC-GCPDS/Documentos/data/Rayos.pkl")

    vegetacion = pd.read_pickle("C:/Users/lucas/OneDrive - Universidad Nacional de Colombia/PC-GCPDS/Documentos/data/Vegetacion.pkl")

    return trafos, apoyos, switches, redmt, super_eventos, descargas, vegetacion

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
                    top: 15px; right: 10px; width: 110px; height: 160px; 
                    background-color: white; border:2px solid black; 
                    z-index:9999; font-size:12px; padding: 8px; opacity: 0.7;">
            <i style="background-color:blue; width: 15px; height: 15px; display: inline-block;"></i> Apoyos<br>
            <i style="background-color:green; width: 15px; height: 15px; display: inline-block;"></i> Trafos<br>
            <i style="background-color:purple; width: 15px; height: 15px; display: inline-block;"></i> Switches<br>
            <i style="background-color:black; width: 15px; height: 15px; display: inline-block;"></i> Red MT<br>
            <i style="background-color:yellow; width: 15px; height: 15px; display: inline-block;"></i> CR baja<br>
            <i style="background-color:orange; width: 15px; height: 15px; display: inline-block;"></i> CR media<br>
            <i style="background-color:red; width: 15px; height: 15px; display: inline-block;"></i> CR alta<br>
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
                    color = 'yellow'
                case 1:
                    color = 'orange'
                case _:
                    color = 'red'
            if len(polygon_puntos) != 1:
                folium.Polygon(locations=polygon_puntos, color=color, fill=True, fill_opacity=0.3, weight=0).add_to(mapa)

            puntos_tramos = eval(df['TRAMOS_AGUAS_ABAJO'].values[0])
            nodos_aguas_abajo.update(puntos_tramos)
            
            puntos_verdes = eval(df['EQUIPOS_PUNTOS'].values[0]) 
            puntos_equipos.extend(puntos_verdes)

        codes = [[], [], [], []]

        for _, row in redmt_seleccionado.iterrows():
            point1 = (row["LATITUD"], row["LONGITUD"])
            point2 = (row["LATITUD2"], row["LONGITUD2"])
            
            if (point1 in nodos_aguas_abajo and point2 in nodos_aguas_abajo):
                color = 'black'
                codes[0].append(row['CODE']) 
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
                codes[1].append(row['CODE'])
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
                codes[2].append(row['CODE'])
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
                codes[3].append(row['CODE'])
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
    
        return mapa_html, codes
