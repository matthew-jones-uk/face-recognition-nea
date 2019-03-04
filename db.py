import sqlite3
from flask import current_app, g

def get_db():
    """Return database. Connect if not already defined."""
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config('DATABASE')
        )
    return g.db

def close_db():
    """Function to close database on Flask application shutdown."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    """Run SQL schema on database init, if database is not available
    this will create it
    """
    db = get_db()
    with current_app.open_resource('schema.sql') as schema:
        db.executescript(schema.read().decode('utf8'))

def init_app(app):
    app.teardown_appcontext(close_db)
