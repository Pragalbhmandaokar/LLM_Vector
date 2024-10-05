# %% module imports
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma

# %% set up data paths
VECTORSTORE_PATH = "./../data/vectorstores/BAAI/bge-small-en-v1.5"

# %% set up the embedding function
bge_model = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceBgeEmbeddings(model_name=bge_model)

# %% load the vector store
db = Chroma(persist_directory=str(VECTORSTORE_PATH), embedding_function=embeddings)

# %% put db in a retriever mode
search_type = "similarity"
search_kwargs = {
    "k": 10,
}
similarity_retriever = db.as_retriever(
    search_type=search_type,
    search_kwargs=search_kwargs,
)

# %% fire off a single query
query = "I'm looking for a highly engaging trainer to deliver an in-person course on business communication in Boston. The ideal trainer should be considerate and patient, able to handle large groups and make the sessions interactive and fun."

# %% fire off a single query (optional filter)
search_kwargs = {
    "k": 10,
    # 'filter': {"year_trading_started": 2023}
}

similarity_retriever_filter = db.as_retriever(
    search_type=search_type,
    search_kwargs=search_kwargs,
)

results = similarity_retriever_filter.invoke(query)
for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print(result.page_content)
    print(result.metadata['provider_name'])
    print(result.metadata['trainer_name'])
    