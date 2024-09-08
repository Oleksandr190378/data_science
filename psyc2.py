import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import numpy as np
from action_1 import df_new

connection_string = "dbname=postgres user=postgres password=mysecretpassword host=localhost"

batch_size = 1000

def insert_batch(cur, data_list):
    execute_values(cur, "INSERT INTO embeddings (content, tokens, embedding) VALUES %s", data_list)

data_list = [(row['content'], int(row['tokens']), np.array(row['embeddings'])) for index, row in df_new.iterrows()]



# Insert data in batches
with psycopg2.connect(connection_string) as conn:
    register_vector(conn)
    with conn.cursor() as cur:
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            insert_batch(cur, batch)
            conn.commit()

# Create an index on the embeddings table for faster retrieval
with psycopg2.connect(connection_string) as conn:
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute("CREATE INDEX  ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);")
        conn.commit()
 
        
# Verify the number of inserted records and print the first record
with psycopg2.connect(connection_string) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) as cnt FROM embeddings;")
        num_records = cur.fetchone()[0]
        print("Number of vector records in table: ", num_records, "\n")

        cur.execute("SELECT * FROM embeddings LIMIT 1;")
        records = cur.fetchall()
        print("First record in table: ", records)