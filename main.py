from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import base64
import random
import re
import mysql.connector
import os

from algo_static import static_model
from algo_dynamic import dynamic_model

#Cargar variables de entorno
load_dotenv()

#Inicializar Flask y CORS
app = Flask(__name__)
CORS(app)

#Conexión a base de datos
mysql_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}

# Metodo para convertir imagen a base64
def get_image_as_base64(image_filename):
    with open(f"static/images/{image_filename}", "rb") as image_file:
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        image_file.close()
    return base64_image

# Metodo para convertir video a base64
def get_video_as_base64(video_filename):
    with open(f"static/videos/{video_filename}", "rb") as video_file:
        video_data = video_file.read()
        base64_video = base64.b64encode(video_data).decode('utf-8')
    return base64_video

served_rows = []
available_rows = []

def load_available_rows():
    connection = mysql.connector.connect(**mysql_config)
    cursor = connection.cursor()
    #query = "SELECT titulo, video, definicion, imagen, dinamico, señaID FROM seña WHERE leccionID = 1 AND (dinamico = 0 OR señaID = 10 OR señaID = 10 OR señaID = 13 OR señaID = 30)"
    query = "SELECT titulo, video, definicion, imagen, dinamico, señaID FROM seña"
    cursor.execute(query)
    data = cursor.fetchall()
    
    for item in data:
        available_rows.append({
            'titulo': item[0],
            'video64': item[1],
            'definicion': item[2],
            'imagen64': item[3],
            'dinamico': item[4],
            'señaID': item[5]
        })

load_available_rows()

# Ruta para obtener frames de la camara (estaticos)
@app.route('/process_frame', methods=['POST'])
def process_frame():
    #try:
        frame = request.json.get('frame')
        palabra = request.json.get('palabra')
        id = request.json.get('palabraId')
        connection = mysql.connector.connect(**mysql_config)
        cursor = connection.cursor()
        
        query = "SELECT pulgar, indice, medio, anular, meñique FROM umbrales WHERE señaID = %s"
        cursor.execute(query, (id,))
        data = cursor.fetchall()

        if frame.startswith('data:'):
            frame = re.sub('^data:image/.+;base64,', '', frame)
        respuesta = static_model(frame, palabra, data[0][0], data[0][1], data[0][2], data[0][3], data[0][4])
        with open('datos_recibidos.txt', 'w') as archivo:
            archivo.write(str(respuesta[1]))
        return jsonify(respuesta[0])
    # except Exception as e:
    #     return jsonify({"error": str(e)})
    
# Ruta para obtener frames de la camara (dinamicos)
@app.route('/process_frame_dynamic', methods=['POST'])
def process_frame_dynamic():
    #try:
        frames = request.json.get('frames')
        palabra = request.json.get('palabra')
        pasos = request.json.get('numSteps')
        id = request.json.get('palabraId')
        connection = mysql.connector.connect(**mysql_config)
        cursor = connection.cursor()
        
        query = "SELECT umbral FROM umbralesPose WHERE señaID = %s"
        cursor.execute(query, (id,))
        data = cursor.fetchall()
        if len(data) == 0:
            respuesta = dynamic_model(frames, palabra, pasos)
        else:
            respuesta = dynamic_model(frames, palabra, pasos, data[0][0])
            #print(respuesta)
        return jsonify(respuesta)
    # except Exception as e:
    #     print("e 106 main")
    #     print(e)
    #     return jsonify({"error": str(e)})

# Ruta para obtener frames de la camara (dinamico - mano)
@app.route('/process_frame_dynamic_hand', methods=['POST'])
def process_frame_dynamic_hand():
    try:
        frame = request.json.get('frame')
        palabra = request.json.get('palabra')
        id = request.json.get('palabraId')
        index = request.json.get('index')
        connection = mysql.connector.connect(**mysql_config)
        cursor = connection.cursor()
        query = "SELECT pulgar, indice, medio, anular, meñique FROM umbrales WHERE señaID = %s AND paso = %s"
        cursor.execute(query, (id,index,))
        data = cursor.fetchall()

        if frame.startswith('data:'):
            frame = re.sub('^data:image/.+;base64,', '', frame)
        respuesta = static_model(frame, palabra, data[0][0], data[0][1], data[0][2], data[0][3], data[0][4], index, True)
        with open('datos_recibidos.txt', 'w') as archivo:
            archivo.write(str(respuesta[1]))
        return jsonify(respuesta[0])
    except Exception as e:
        return jsonify({"error": str(e)})

# Ruta para obtener todas las lecciones (Con titulo e imagen)
@app.route('/aprende', methods=['GET'])
def aprende():
    connection = mysql.connector.connect(**mysql_config)
    cursor = connection.cursor()
    
    cursor.execute("SELECT * FROM leccion")
    columns = [column[0] for column in cursor.description]
    data = [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    for row in data:
        if 'imagen' in row:
            imagen64 = row['imagen']
            imagen64 = get_image_as_base64(imagen64)
            row['imagen64'] = imagen64
    

    cursor.close()
    connection.close()
    return jsonify(data)

# 2da Opción - Ruta para obtener cierta palabra de cierta lección
@app.route('/lecciones/<int:id_leccion>/<int:id_seccion>', methods=['GET'])
def get_lecciones(id_leccion, id_seccion):
    id_seccion = id_seccion - 1
    connection = mysql.connector.connect(**mysql_config)
    cursor = connection.cursor()
    
    query = "SELECT titulo, imagen, video, definicion, dinamico FROM seña WHERE leccionID = %s ORDER BY señaID LIMIT %s, 1"
    cursor.execute(query, (id_leccion, id_seccion))
    data = cursor.fetchall()
    
    if len(data) == 0:
        return jsonify({"error": "Sección no encontrada."}), 404
    
    if data[0][4] == 0:
        howTo = get_image_as_base64(data[0][2])
    else:
        howTo = get_video_as_base64(data[0][2])
    
    # TODO Convertir video a base64
    item = {
        'titulo': data[0][0],
        'imagen64': get_image_as_base64(data[0][1]),
        'video64': howTo,
        'definicion': data[0][3],
        'dinamico': data[0][4]
    }
    
    return jsonify(item)

## Array de todas las secciones por lección
@app.route('/<int:id_leccion>', methods=['GET'])
def get_todas_las_secciones(id_leccion):
    connection = mysql.connector.connect(**mysql_config)
    cursor = connection.cursor()
    
    query = """
    SELECT titulo, imagen, video, definicion, señaID, (@counter := @counter + 1) AS counter
    FROM (SELECT @counter := 0) AS init
    JOIN seña
    WHERE leccionID = %s;
    """
    cursor.execute(query, (id_leccion,))
    data = cursor.fetchall()
    
    if len(data) == 0:
        return jsonify({"error": "Lección no encontrada."}), 404
    
    # Convertir imagen a base64 TODO Convertir video a base64
    items = []
    for item in data:
        items.append({
            'titulo': item[0],
            'imagen64': get_image_as_base64(item[1]),
            'video64': item[2],
            'definicion': item[3],
            'leccionID': item[4],
            'counter': int(item[5])
        })
        
    return jsonify(items)
    
# Ruta para "Practica" aleatoriamente elige una seccion de todas las lecciones
# TODO Incorporar base de datos
@app.route('/random', methods=['GET'])
def seccion_random():
    if not available_rows:
        load_available_rows()
        served_rows.clear()

    random_index = random.randrange(len(available_rows))
    selected_row = available_rows.pop(random_index)
    
    selected_row['pasos'] = 0
    if 'dinamico' in selected_row:
        if selected_row['dinamico'] == 1:
            connection = mysql.connector.connect(**mysql_config)
            cursor = connection.cursor()
            query = "SELECT pasos FROM momentos WHERE señaID = %s"
            cursor.execute(query, (selected_row['señaID'],))
            data = cursor.fetchall()
            selected_row['pasos'] = data[0][0]	

            query = "SELECT COUNT(*) FROM umbrales WHERE señaID = %s"
            cursor.execute(query, (selected_row['señaID'],))
            data = cursor.fetchall()
            selected_row['cambios'] = data[0][0]	

    if "mp4" not in selected_row['video64']:
        howTo = get_image_as_base64(selected_row['video64'])
    else:
        howTo = get_video_as_base64(selected_row['video64'])

    selected_row['imagen64'] = get_image_as_base64(selected_row['imagen64'])
    selected_row['video64'] = howTo
    # TODO hacerlo no hardcodeado para a xd

    served_rows.append(selected_row)

    return jsonify(selected_row)

@app.route('/')
@app.route('/<path:path>')
def catch_all(path='index.html'):
    return render_template(path)
        

if __name__ == '__main__':
    app.run(host="34.125.128.49",debug=True, port=5000)
