# %% module imports
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from pathlib import Path
from sklearn.metrics import ndcg_score
import pandas as pd
import numpy as np


# %% set up data paths
root_path = Path(__file__).resolve().parents[1]
data_path = (
    root_path
    / "data"
    / "vectorstores"
    / "training_providers_merged"
    / "BAAI"
    / "bge-small-en-v1.5"
)

# %% set up the embedding function
bge_model = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceBgeEmbeddings(model_name=bge_model)

# %% load the vector store
db = Chroma(persist_directory=str(data_path), embedding_function=embeddings)

# %%
documents = db.get()["documents"]   # List of document texts
metadatas = db.get()["metadatas"]   # List of metadata dictionaries

# %% use bm25 retriever
number_to_retrieve = 10
retriever = BM25Retriever.from_texts(documents, metadatas=metadatas, k=number_to_retrieve)
# %% fire off a single query
query = "I’m seeking a dynamic and passionate trainer for a two-week course on strategic management in London. The firm should have their own premises and the trainer should be known for making complex concepts easy to grasp."
results = retriever.invoke(query)


# %%
for result in results:
    name = result.metadata.get('name')
    page_content = result.page_content
    print(name)
    # print(f"Page Content: {page_content}")
    # # print("\n" + "="*50 + "\n")


#%% load queries
QUERIES_PATH = "../data/raw/queries.xlsx"

queries = pd.read_excel(QUERIES_PATH)
# strip '"' from 'training_providers' and 'query' column
queries['query'] = queries['query'].str.replace('"','')
queries['training_providers'] = queries['training_providers'].str.replace('"','')

# cast 'training_providers' to list
queries['training_providers'] = queries['training_providers'].apply(lambda x: x.split(","))

#%% Helper function to get binary relevance scores
def get_relevance_scores(suggested_providers, expected_providers):
    return [1 if provider in expected_providers else 0 for provider in suggested_providers]


ndcg_scores = []
# Iterate through each query and calculate the NDCG score
for i, query in enumerate(queries['query']):
    results = retriever.invoke(query)
    suggested_providers = []
    
    for result in results:
        suggested_providers.append(result.metadata.get('name'))
    
    # Get the expected providers
    expected_providers = queries['training_providers'][i]
    
    # Truncate the longer list to match the size of the shorter list
    min_length = min(len(suggested_providers), len(expected_providers))
    suggested_providers = suggested_providers[:min_length]
    expected_providers = expected_providers[:min_length]
    
    # Map suggested providers to relevance scores (binary)
    relevance_scores = get_relevance_scores(suggested_providers, expected_providers)
    
    # Since ndcg_score requires 2D arrays, reshape the relevance scores
    relevance_scores = np.array(relevance_scores).reshape(1, -1)
    
    # The ideal relevance (1 for each expected provider, 0 for the rest)
    ideal_relevance_scores = np.array([1 if provider in suggested_providers else 0 for provider in expected_providers]).reshape(1, -1)
    
    # Calculate NDCG score
    ndcg = ndcg_score(ideal_relevance_scores, relevance_scores)
    
    # Append the NDCG score to the list
    ndcg_scores.append(ndcg)
    
    print(f"Query: {i+1}")
    print(f"Suggested Providers: {suggested_providers}")    
    print(f"Expected Providers: {sorted(expected_providers)}")
    print(f"NDCG Score: {ndcg:.4f}")

# %% give the average ndcg score
average_ndcg = np.mean(ndcg_scores)
print(f"Average NDCG Score: {average_ndcg:.4f}")


# %%
