from langchain_huggingface import HuggingFaceEmbeddings

# Khởi tạo model embedding
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# List text cần embed
texts = ["Xin chào bạn", "Hôm nay trời đẹp", "Chúng ta đi chơi nhé"]

# Tạo embedding cho từng văn bản
embeddings = embedding_model.embed_documents(texts)

# In kết quả embedding
for idx, emb in enumerate(embeddings):
    print(f"Văn bản {idx+1}: {texts[idx]}")
    print(f"Vector embedding (kích thước {len(emb)}):\n{emb}\n")
