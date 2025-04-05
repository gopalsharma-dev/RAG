import pickle
from rag_utils import load_pdf_chunks, embed_chunks

chunks = load_pdf_chunks("DBS 2.pdf")
embeddings = embed_chunks(chunks)

import faiss
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

faiss.write_index(index, "vector.index")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
