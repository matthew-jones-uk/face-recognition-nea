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