from flask import Flask, jsonify, request
from flask_cors import CORS
import base64
import random
import re

# Importar los arrays de las lecciones de lecciones.py

from lecciones import lecciones
from lecciones_temp import abecedario
from lecciones import saludos
from lecciones_temp import secciones

# Array de los IDs de todas las secciones
from lecciones_temp import secciones_random

from algo_static import static_model

app = Flask(__name__)
CORS(app)

# Metodo para convertir imagen a base64
def get_image_as_base64(image_filename):
    with open(f"static/images/{image_filename}", "rb") as image_file:
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        image_file.close()
    return base64_image

# Metodo para convertir video a base64
""" def get_video_as_base64(video_filename):
    with open(f"static/videos/{video_filename}", "rb") as video_file:
        video_data = video_file.read()
        base64_video = base64.b64encode(video_data).decode('utf-8')
    return base64_video """


# Ruta para obtener frames de la camara
@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        frame = request.json.get('frame')
        palabra = request.json.get('palabra')
        #fingers_done = request.json.get('fingersDone')
        #print(palabra)
        if frame.startswith('data:'):
            frame = re.sub('^data:image/.+;base64,', '', frame)
        respuesta= static_model(frame, palabra)
        print(respuesta[1])
        with open('datos_recibidos.txt', 'w') as archivo:
            archivo.write(str(respuesta[1]))
        return jsonify(respuesta[0])
    except Exception as e:
        return jsonify({"error": str(e)})

# Ruta para obtener todas las lecciones (Con titulo e imagen)
@app.route('/aprende', methods=['GET'])
def aprende():
    for item in lecciones:
        item['imagen64'] = get_image_as_base64(item['imagen'])
    return jsonify(lecciones)

# 1ra Opción

@app.route('/abecedario/<int:id>', methods=['GET'])
def get_abecedario(id):
    for item in abecedario:
        if item['id'] == id:
            return jsonify(item)
    return jsonify({"error": "Letra no encontrada."}), 404

""" @app.route('/numero/<int:id>', methods=['GET'])
def getNumeros(id):
    for item in numeros:
        if item['id'] == (id):
            return jsonify(item)
    return jsonify({"error": "Número no encontrado."}), 404 """

@app.route('/saludo/<int:id>', methods=['GET'])
def get_saludos(id):
    for item in saludos:
        if item['id'] == (id):
            return jsonify(item)
    return jsonify({"error": "Saludo no encontrado."}), 404

# 2da Opción - Ruta para obtener cierta sección de cierta lección
@app.route('/lecciones/<int:id_leccion>/<int:id_seccion>', methods=['GET'])
def get_lecciones(id_leccion, id_seccion):
    if id_leccion == 1:
        for item in abecedario:
            if item['id'] == id_seccion:
                item['imagen64'] = get_image_as_base64(item['imagen'])
                item['video64'] = get_image_as_base64(item['video'])
                """ item['video'] = get_video_as_base64(item['video_filename']) """
                return jsonify(item)
        return jsonify({"error": "Letra no encontrada."}), 404
    elif id_leccion == 2:
        for item in saludos:
            if item['id'] == id_seccion:
                item['imagen64'] = get_image_as_base64(item['imagen'])
                """ item['video'] = get_video_as_base64(item['video_filename']) """
                return jsonify(item)
        return jsonify({"error": "Saludo no encontrado."}), 404
    else:
        return jsonify({"error": "Lección no encontrada."}), 404
    

## Array de todas las secciones por lección
@app.route('/<int:id_leccion>', methods=['GET'])
def get_todas_las_secciones(id_leccion):
    if id_leccion == 1:
        return jsonify(abecedario)
    if id_leccion == 2:
        return jsonify(saludos)
    else:
        return jsonify({"error": "Lección no encontrada."}), 404
    
# Ruta para "Practica" aleatoriamente elige una seccion de todas las lecciones
@app.route('/random', methods=['GET'])
def seccion_random():
    global secciones_random
    if len(secciones_random) == 0:
        secciones_random = [seccion['id'] for seccion in secciones]
        
    id_aleatorio = random.choice(secciones_random)
    secciones_random.remove(id_aleatorio)
        
    for item in secciones:
        if item['id'] == id_aleatorio:
            item['imagen64'] = get_image_as_base64(item['imagen'])
            """ item['video'] = get_video_as_base64(item['video_filename']) """
            return jsonify(item)
        

if __name__ == '__main__':
    app.run(debug=True, port=4000)