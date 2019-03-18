from os.path import join, isdir
from os import makedirs, listdir, remove
from time import sleep, time
from base64 import b64encode
from uuid import uuid4
import pickle
import threading
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

db_handler = db.ThreadHandler(join(INSTANCE, DATABASE_FILENAME))

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
    ''' Function for the route to get the face id in json format'''
    with db_handler:
        random_image_record = db_handler.cursor.execute(
            'SELECT * FROM images WHERE active = 1 ORDER BY RANDOM() LIMIT 1'
        ).fetchone()
    if random_image_record is None:
        response = jsonify(status=1, id='')
        response.status_code = 404
        return response
    else:
        response = jsonify(status=0, id=random_image_record[0])
        response.status_code = 200
        return response

@app.route('/getFaceImage', methods=['GET'])
def get_face_image():
    ''' Function for the route to get the face image in base64 json when given id'''
    try:
        given_id = request.headers.get('id')
    except KeyError:
        # if there's no id, return status 400 bad request with invalid json data json status code
        response = jsonify(status=4, id='')
        response.status_code = 400
        return response
    with db_handler:
        record = db_handler.cursor.execute(
            'SELECT * FROM images WHERE id = "{}"'.format(given_id)
        ).fetchone()
    if record is None:
        # if no record is found, return status 404 not found with invalid id json status code
        response = jsonify(status=2, id=given_id)
        response.status_code = 404
        return response
    elif record[7] == 0:
        # if no longer active in database, return status 410 gone with invalid id json status code
        response = jsonify(status=2, id=given_id)
        response.status_code = 410
        return response
    else:
        # set filename and open file
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
        # return with status 200 OK and success status code
        response = jsonify(status=0, id=given_id,
                           probability=probability, image=base64_image)
        response.status_code = 200
        return response


@app.route('/giveVote', methods=['POST'])
def give_vote():
    ''' Function for the route to give a vote on whether or not a face '''
    json = request.get_json(silent=True)
    if json is None:
        # check for json data, if not found then return status 400 bad request with
        # 'invalid json' json status code
        response = jsonify(status=4)
        response.status_code = 400
        return response
    elif 'id' not in json or 'vote' not in json:
        # if id/vote not found in json, then return status 400 bad request with
        # 'invalid json' json status code
        response = jsonify(status=4)
        response.status_code = 400
        return response
    else:
        given_id, vote = json['id'], json['vote']
        # check if id is valid in database
        with db_handler:
            record = db_handler.cursor.execute(
                'SELECT * FROM images WHERE id = "{}"'.format(given_id)
            ).fetchone()
        if record is None:
            # if id is invalid return status 400 not found with invalid id json status code
            response = jsonify(status=2, id=given_id)
            response.status_code = 404
            return response
        # consider the vote according to json data and add to database
        if vote:
            with db_handler:
                db_handler.cursor.execute('''
                    UPDATE images
                    SET positive_votes = positive_votes + 1
                    WHERE id = "{}"
                '''.format(given_id))
        else:
            with db_handler:
                db_handler.cursor.execute('''
                    UPDATE images
                    SET negative_votes = negative_votes + 1
                    WHERE id = "{}"
                '''.format(given_id))
        # return status code 200 OK and success json status code
        response = jsonify(status=0, id=given_id)
        response.status_code = 200
        return response

def get_model():
    '''This function retrieves the latest saved version of the detection model based on filenames.
    Returns:
        face_detection.Model: Detection model
    '''
    # retrieve all model files and calculate latest based on number after root name
    all_files = listdir(INSTANCE)
    model_files = dict()
    for test_file in all_files:
        if test_file.startswith(ROOT_MODEL_FILENAME):
            try:
                number = int(''.join([int(s) for s in test_file if s.isdigit()]))
            except ValueError:
                # if no number can be found, default to zero
                number = 0
            model_files[test_file] = number
    # get highest value in dictionary
    values = list(model_files.values())
    keys = list(model_files.keys())
    model = keys[values.index(max(values))]
    # load the model as a face_detection.Model using pickle
    return pickle.load(join(INSTANCE, model))

def detect_faces(image):
    '''This founction uses the face_detection module to find any faces in an image
    Args:
        image (Image): Skimage image array format
    Returns:
        list: Returns a list of face_detection.Face objects
    '''
    model = get_model()
    found_faces = face_detection.find_all_face_boxes(image, model)
    for face in found_faces:
        face.find_face_image(image)
    return found_faces

def new_image_detector(db_handler):
    '''Checks the testing image directory for any new images then processes and adds to database.
       Should be run in own process due to CPU and IO heavy and blocking nature.
    '''
    while True:
        # get all files in directory
        files = listdir(join(INSTANCE, DATABASE_TESTING_IMAGES_DIRECTORY))
        faces = list()
        # for every file in directory, detect faces and remove file
        for image_file in files:
            faces = faces + detect_faces(face_detection.load_image(join(INSTANCE, image_file)))
            remove(join(INSTANCE, DATABASE_TESTING_IMAGES_DIRECTORY, image_file))
        # for any detected face generate unique id, save file and add to database
        for face in faces:
            checking = True
            while checking:
                unique_id = uuid4()
                with db_handler:
                    record = db_handler.cursor.execute('''
                                SELECT * FROM images WHERE id = "{}"'''.format(unique_id)).fetchone()
                if not record:
                    checking = False
            filename = unique_id+'.png'
            # calculate needed votes out of 10
            needed_votes = 10 - round(face.probability*100, -1)/10
            with db_handler:
                db_handler.cursor.execute('''
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
                '''.format(unique_id, filename, time(), face.probability, needed_votes))
            imsave(filename, face.face_image)
        sleep(DETECTOR_TIMEOUT)

def run():
    ''' Main application function '''
    # this will check if required directories exist and if they don't, create them
    if not isdir(INSTANCE):
        makedirs(INSTANCE)
    if not isdir(join(INSTANCE, DATABASE_IMAGES_DIRECTORY)):
        makedirs(join(INSTANCE, DATABASE_IMAGES_DIRECTORY))
    if not isdir(join(INSTANCE, DATABASE_TESTING_IMAGES_DIRECTORY)):
        makedirs(join(INSTANCE, DATABASE_TESTING_IMAGES_DIRECTORY))
    # creates database if not already made
    with db_handler:
        db_handler.cursor.executescript('''
        CREATE TABLE IF NOT EXISTS images (
                    id text UNIQUE PRIMARY KEY NOT NULL, 
                    filename text UNIQUE NOT NULL,
                    start_date integer,
                    probability real,
                    positive_votes integer DEFAULT 0,
                    negative_votes integer DEFAULT 0,
                    needed_votes integer NOT NULL,
                    active integer DEFAULT 1
            )     
        ''')    
    app.run(debug=True, host='0.0.0.0')

if __name__ == '__main__':
    run()
