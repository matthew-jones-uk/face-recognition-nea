from os.path import join, isdir
from os import makedirs, listdir, rename, remove
from time import sleep, time
from base64 import b64encode
from uuid import uuid4
import pickle
from flask import Flask, jsonify, request
from skimage.io import imsave
import db
import face_detection

INSTANCE = 'instance'
DATABASE_FILENAME = 'db.sqlite'
DATABASE_IMAGES_DIRECTORY = 'images'
DATABASE_TESTING_IMAGES_DIRECTORY = 'testing_images'
ROOT_MODEL_FILENAME = 'model.sav'

DETECTOR_TIMEOUT = 60

app = Flask(__name__)

'''
Error Codes:
0 - Success
1 - Database Empty
2 - Invalid ID
3 - Invalid vote
4 - Invalid JSON data
'''

_temp_ids = dict()

@app.route('/getFaceId', methods=['GET'])
def get_face_id():
    connection = db.get_db(join(INSTANCE, DATABASE_FILENAME))
    random_image_record = connection.execute(
        'SELECT * FROM images WHERE active = 1 ORDER BY RANDOM() LIMIT 1'
    ).fetchone()
    connection.close()
    if random_image_record is None:
        response = jsonify(status=1, id="")
        response.status_code = 404
        return response
    else:
        response = jsonify(status=0, id=random_image_record[0])
        response.status_code = 200
        return response

@app.route('/getFaceImage', methods=['GET'])
def get_face_image():
    try:
        given_id = request.headers.get('id')
    except KeyError:
        response = jsonify(status=4, id="")
        response.status_code = 400
        return response
    connection = db.get_db(join(INSTANCE, DATABASE_FILENAME))
    record = connection.execute(
        'SELECT * FROM images WHERE id = "{}"'.format(given_id)
    ).fetchone()
    connection.close()
    if record is None:
        response = jsonify(status=2, id=given_id)
        response.status_code = 404
        return response
    elif record[7] == 0:
        response = jsonify(status=2, id=given_id)
        response.status_code = 410
        return response
    else:
        filename = record[1]
        filename = join(INSTANCE, DATABASE_IMAGES_DIRECTORY, filename)
        try:
            with open(filename, 'rb') as image:
                base64_image = b64encode(image.read())
                base64_image = base64_image.decode('ascii')
        except FileNotFoundError:
            response = jsonify(status=4, id=given_id)
            response.status_code = 500
            return response
        probability = record[3]
        response = jsonify(status=0, id=given_id,
                           probability=probability, image=base64_image)
        response.status_code = 200
        return response


@app.route('/giveVote', methods=['POST'])
def give_vote():
    json = request.get_json(silent=True)
    if json is None:
        response = jsonify(status=4)
        response.status_code = 400
        return response
    elif 'id' not in json or 'vote' not in json:
        response = jsonify(status=4)
        response.status_code = 400
        return response
    else:
        given_id, vote = json['id'], json['vote']
        connection = db.get_db(join(INSTANCE, DATABASE_FILENAME))
        record = connection.execute(
            'SELECT * FROM images WHERE id = "{}"'.format(given_id)
        ).fetchone()
        if record is None:
            connection.close()
            response = jsonify(status=2, id=given_id)
            response.status_code = 404
            return response
        if vote:
            connection.execute("""
                UPDATE images
                SET positive_votes = positive_votes + 1
                WHERE id = "{}"
            """.format(given_id))
            connection.commit()
        else:
            connection.execute("""
                UPDATE images
                SET negative_votes = negative_votes + 1
                WHERE id = "{}"
            """.format(given_id))
            connection.commit()
        connection.close()
        response = jsonify(status=0, id=given_id)
        response.status_code = 200
        return response

def get_model():
    # retrieve all model files and calculate latest based on number after root name
    all_files = listdir(INSTANCE)
    model_files = dict()
    for test_file in all_files:
        if test_file.startswith(ROOT_MODEL_FILENAME):
            try:
                number = int(''.join([int(s) for s in test_file if s.isdigit()]))
            except ValueError:
                number = 0
            model_files[test_file] = number
    # get highest value in dictionary
    values = list(model_files.values())
    keys = list(model_files.keys())
    model = keys[values.index(max(values))]
    return pickle.load(join(INSTANCE, model))

def detect_faces(image):
    model = get_model()
    found_faces = face_detection.find_all_face_boxes(image, model)
    for face in found_faces:
        face.find_face_image(image)
    return found_faces

def new_image_detector():
    while True:
        files = listdir(join(INSTANCE, DATABASE_TESTING_IMAGES_DIRECTORY))
        if len(files) == 0:
            continue
        faces = list()
        for image_file in files:
            faces = faces + detect_faces(face_detection.load_image(join(INSTANCE,
                        image_file)))
            remove(join(INSTANCE, DATABASE_TESTING_IMAGES_DIRECTORY, image_file))
        for face in faces:
            connection = db.get_db(join(INSTANCE, DATABASE_FILENAME))
            checking = True
            while checking:
                unique_id = uuid4()
                record = connection.execute('SELECT * FROM images WHERE id = {}'.format(unique_id)).fetchone()
                if len(record) == 0:
                    checking = False
            filename = unique_id+'.png'
            needed_votes = 10 - round(face.probability*100, -1)/10
            connection.execute("""
                INSERT INTO images (
                    id,
                    filename,
                    start_date,
                    probability,
                    needed_votes
                ) VALUES (
                    {},
                    {},
                    {},
                    {},
                    {}
                )
            """.format(unique_id, filename, time(), face.probability, needed_votes))
            imsave(filename, face.face_image)
        sleep(DETECTOR_TIMEOUT)

def run():
    if not isdir(INSTANCE):
        makedirs(INSTANCE)
    if not isdir(join(INSTANCE, DATABASE_IMAGES_DIRECTORY)):
        makedirs(join(INSTANCE, DATABASE_IMAGES_DIRECTORY))
    if not isdir(join(INSTANCE, DATABASE_TESTING_IMAGES_DIRECTORY)):
        makedirs(join(INSTANCE, DATABASE_TESTING_IMAGES_DIRECTORY))
    db.init_db(join(INSTANCE, DATABASE_FILENAME))
    app.run(debug=True, host="0.0.0.0")

if __name__ == '__main__':
    run()
