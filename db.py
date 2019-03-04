import sqlite3
from time import time
from uuid import uuid4

def get_db(database_path):
    """Return database"""
    connection = sqlite3.connect(database_path)
    return connection

def init_db(database_path):
    """Run SQL schema on database init, if database is not available
    this will create it
    """
    db = sqlite3.connect(database_path)
    db.executescript("""
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
    """)
    db.close()
