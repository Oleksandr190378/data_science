#  PostgreSQL
FROM postgres:latest

RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql-server-dev-all \
    git

RUN git clone https://github.com/pgvector/pgvector.git /tmp/pgvector

WORKDIR /tmp/pgvector

RUN make && make install

RUN apt-get clean && rm -rf /var/lib/apt/lists/*