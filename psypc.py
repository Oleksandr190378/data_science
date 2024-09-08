import psycopg2
from pgvector.psycopg2 import register_vector
connection_string = "dbname=postgres user=postgres password=mysecretpassword host=localhost"
conn = psycopg2.connect(connection_string)
cur = conn.cursor()
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
conn.commit()

register_vector(conn)
table_create_command = """
CREATE TABLE embeddings (
    id bigserial primary key, 
    content text,
    tokens integer,
    embedding vector(768)
);
"""


cur.execute(table_create_command)
cur.close()
conn.commit()

