import face_detection
import db
from flask import Flask, jsonify
from os.path import join
from os import makedirs
from uuid import uuid4

app = Flask(__name__)
app.config.from_mapping(
    DATABASE = join(app.instance_path,'db.sqlite')
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
2 - Invalid UUID
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
        'SELECT * FROM table'
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

@app.route('/getFaceImage', methods=['POST'])
def get_face_image():
    ...

@app.route('/giveVote', methods=['POST'])
def give_vote():
    ...

if __name__ == '__main__':
    app.run(debug=True)