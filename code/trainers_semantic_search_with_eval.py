# %% module imports
import pandas as pd
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma

# %% set up data paths
VECTORSTORE_PATH = "./../data/vectorstores/BAAI/bge-small-en-v1.5"
QUERIES_PATH = "./../data/queries.xlsx"

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

# %% load queries
queries = pd.read_excel(QUERIES_PATH)
# strip '"' from 'training_providers' and 'query' column
queries['query'] = queries['query'].str.replace('"','')
queries['training_providers'] = queries['training_providers'].str.replace('"','')

# cast 'training_providers' to list
queries['training_providers'] = queries['training_providers'].apply(lambda x: x.split(",")) 

# %% for each query in test queries dataframe, retrieve the top k similar training providers
for i, query in enumerate(queries['query']):
    results = similarity_retriever.invoke(query)
    suggested_providers = []
    for result in results:
        suggested_providers.append(result.metadata['provider_name'])
    print(f"Query: {i+1}")
    print(f"Suggested Providers: {suggested_providers}")    
    print(f"Expected Providers: {sorted(queries['training_providers'][i])}")
# %%
