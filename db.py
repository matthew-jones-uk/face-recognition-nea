import sqlite3
from flask import current_app, g

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config('DATABASE')
        )
    return g.db

def close_db():
    db = g.pop('db',None)
    if db is not None:
        db.close()

def init_db():
    # check for tables and create them
    ...

def init_app(app):
    app.teardown_appcontext(close_db)