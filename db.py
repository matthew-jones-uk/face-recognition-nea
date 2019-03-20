import sqlite3
from threading import Lock

class ThreadHandler():
    '''Object to handle thread-safe database interactions. It's lockable so calls wait to acquire
       the lock.
    Returns:
        self: Returns itself which contains a connection and cursor for database queries.
    '''
    def __init__(self, database_path):
        # Create lock for object in object initialisation.
        self.lock = Lock()
        # Create to SQL database. We can keep the connection open as nothing else is accessing.
        self.connection = sqlite3.connect(database_path, check_same_thread=False)
        self.connection.executescript('''
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
        # Create cursor variable.
        self.cursor = None
    def __enter__(self):
        # When object is called, try to acquire lock. If unavailable wait until possible.
        self.lock.acquire()
        # Create cursor to interact with database.
        self.cursor = self.connection.cursor()
        return self
    def __exit__(self, type, value, traceback):
        # Automatically commit any changes to database on competion.
        self.connection.commit()
        # If a cursor exists then close it.
        if self.cursor is not None:
            self.cursor.close()
            self.cursor = None
        # Release lock for other threads.
        self.lock.release()
