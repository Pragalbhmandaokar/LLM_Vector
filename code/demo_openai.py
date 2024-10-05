#%% import libraries
import pandas as pd
import numpy as np
from pickle import load
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from sklearn.metrics import ndcg_score
from langchain import embeddings

# %% data paths
root_path = Path(__file__).resolve().parents[1]
data_path = root_path / "data" / "pickles"

# %% load docs
input_filename = "training_providers_merged.pkl"
with open(data_path / input_filename, "rb") as f:
    docs = load(f)


#%% vectore store path
model = "openai"
output_dir = root_path / "data" / "vectorstores"
vectorestore_file = output_dir / input_filename.split(".")[0] / model

#%% set up the embedding function
embedding = OpenAIEmbeddings(openai_api_key="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")

vectordb = Chroma.from_documents(documents=docs,
                                 embedding=embedding,
                                 persist_directory=str(vectorestore_file))


# %% retriver
retriever = vectordb.as_retriever()

# %%
query = "I’m seeking a dynamic and passionate trainer for a two-week course on strategic management in London. The firm should have their own premises and the trainer should be known for making complex concepts easy to grasp."
results = retriever.get_relevant_documents(query)

# %% retrive metadata from results
for result in results:
    name = result.metadata.get('name')
    page_content = result.page_content
    print(f"Name: {name}")

#%%
QUERIES_PATH = "../data/raw/queries.xlsx"

# %% load queries
queries = pd.read_excel(QUERIES_PATH)
# strip '"' from 'training_providers' and 'query' column
queries['query'] = queries['query'].str.replace('"','')
queries['training_providers'] = queries['training_providers'].str.replace('"','')

# cast 'training_providers' to list
queries['training_providers'] = queries['training_providers'].apply(lambda x: x.split(","))


# Helper function to get binary relevance scores
def get_relevance_scores(suggested_providers, expected_providers):
    return [1 if provider in expected_providers else 0 for provider in suggested_providers]


ndcg_scores = []
# Iterate through each query and calculate the NDCG score
for i, query in enumerate(queries['query']):
    results = retriever.get_relevant_documents(query)
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
