import sqlite3
from threading import Lock

class ThreadHandler():
    def __init__(self, database_path):
        self.lock = Lock()
        self.connection = sqlite3.connect(database_path, check_same_thread=False)
        self.cursor = None
    def __enter__(self):
        self.lock.acquire()
        self.cursor = self.connection.cursor()
        return self
    def __exit__(self, type, value, traceback):
        self.connection.commit()
        if self.cursor is not None:
            self.cursor.close()
            self.cursor = None
        self.lock.release()
