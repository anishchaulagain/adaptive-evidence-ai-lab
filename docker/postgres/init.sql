-- Extensions required by the platform. Run once at database creation.
CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector: semantic retrieval
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- trigram similarity for keyword search
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
