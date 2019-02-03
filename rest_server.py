import face_detection
import db
from flask import Flask, jsonify, request
from os.path import join
from os import makedirs
from uuid import uuid4
from base64 import b64encode

DATABASE_FILENAME = 'db.sqlite'
DATABASE_IMAGES_DIRECTORY = 'images'

app = Flask(__name__)
app.config.from_mapping(
    DATABASE = join(app.instance_path,DATABASE_FILENAME)
)

try:
    makedirs(app.instance_path)
except OSError:
    pass

db.init_app(app)
db.init_db()

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
    created_id = False
    while not created_id:
        uuid = str(uuid4())
        if uuid not in _temp_ids:
            created_id = True
    random_image_record = db.get_db().execute(
        'SELECT * FROM images'
        'WHERE id IN (SELECT id FROM table ORDER BY RANDOM() LIMIT 1)'
        'AND active = 1'
    ).fetchone()
    if random_image_record is None:
        response = jsonify(status=1, id="")
        response.status_code = 404
        return response
    else:
        _temp_ids[uuid] = random_image_record[0]
        response = jsonify(status=0, id=uuid)
        response.status_code = 200
        return response

@app.route('/getFaceImage', methods=['GET'])
def get_face_image():
    try:
        id = request.headers.get('id')
    except KeyError:
        response = jsonify(status=4, id="")
        response.status_code = 400
        return response
    record = db.get_db().execute(
        'SELECT * FROM images WHERE id="{}"'.format(id)
    ).fetchone()
    if record is None:
        response = jsonify(status=2, id=id)
        response.status_code = 404
        return response
    else:
        if record[7] == 0:
            response = jsonify(status=2, id=id)
            response.status_code = 410
            return response
        else:
            filename = record[1]
            filename = join(app.instance_path,DATABASE_IMAGES_DIRECTORY,filename)
            try:
                with open(filename, 'rb') as image:
                    base64_image = b64encode(image).decode('ascii')
            except FileNotFoundError:
                response = jsonify(status=4, id=id)
                response.status_code = 500
                return response
            probability = record[3]
            response = jsonify(status=0, id=id,
                probability=probability, image=base64_image)
            response.status_code = 200
            return response


@app.route('/giveVote', methods=['POST'])
def give_vote():
    ...

if __name__ == '__main__':
    app.run(debug=True)