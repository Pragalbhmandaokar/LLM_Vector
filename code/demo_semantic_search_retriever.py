# %% module imports
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from pathlib import Path
from sklearn.metrics import ndcg_score
import numpy as np
import pandas as pd

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

# %% put db in a retriever mode
search_type = "similarity"
search_kwargs = {
    "k": 4,
}

similarity_retriever = db.as_retriever(
    search_type=search_type,
    search_kwargs=search_kwargs,
)

# %% fire off a single query to a basic semantic search
query = "I'm looking for a highly engaging trainer to deliver an in-person course on business communication in Boston. The ideal trainer should be considerate and patient, able to handle large groups and make the sessions interactive and fun."
results = similarity_retriever.invoke(query)
for i, result in enumerate(results):
    # print(f"Result {i+1}:")
    print(result.metadata["name"])
    # print(result.page_content)
    print("\n")
    
    
#%% Load queries    
QUERIES_PATH = "../data/raw/queries.xlsx"
queries = pd.read_excel(QUERIES_PATH)
# strip '"' from 'training_providers' and 'query' column
queries['query'] = queries['query'].str.replace('"','')
queries['training_providers'] = queries['training_providers'].str.replace('"','')

# cast 'training_providers' to list
queries['training_providers'] = queries['training_providers'].apply(lambda x: x.split(","))

#%%
# Helper function to get binary relevance scores
def get_relevance_scores(suggested_providers, expected_providers):
    return [1 if provider in expected_providers else 0 for provider in suggested_providers]


ndcg_scores = []
# Iterate through each query and calculate the NDCG score
for i, query in enumerate(queries['query']):
    results = similarity_retriever.invoke(query)
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
    
    
    

# %% fire off a single query with a filter
search_kwargs = {
    "k": 4,
    'filter': {"year_trading_started": 2023}
}
similarity_retriever_filter = db.as_retriever(
    search_type=search_type,
    search_kwargs=search_kwargs,
)
results = similarity_retriever_filter.invoke(query)
for i, result in enumerate(results):
    # print(f"Result {i+1}:")
    # print(result.page_content)
    print(result.metadata["name"])
    print("\n")

# %% specify a retriever based on mmr
search_type = "mmr"
search_kwargs = {"k": 4, "lambda_mult": 0.25}
mmr_retriever = db.as_retriever(
    search_type=search_type,
    search_kwargs=search_kwargs,
)


#%%
ndcg_scores = []
# Iterate through each query and calculate the NDCG score
for i, query in enumerate(queries['query']):
    results = mmr_retriever.invoke(query)
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
