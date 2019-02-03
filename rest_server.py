import face_detection
import db
from flask import Flask, jsonify
from os.path import join
from os import makedirs

app = Flask(__name__)
app.config.from_mapping(
    DATABASE = join(app.instance_path,'db.sqlite')
)

try:
    makedirs(app.instance_path)
except OSError:
    pass

db.init_app(app)

@app.route('/getFaceId')
def get_face_id():
    ...

@app.route('/getFaceImage')
def get_face_image():
    ...

@app.route('/giveVote')
def give_vote():
    ...

if __name__ == '__main__':
    app.run(debug=True)