# %% module imports
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from pathlib import Path
from pickle import load

# %% data paths
root_path = Path(__file__).resolve().parents[1]
data_path = root_path / "data" / "pickles"

# %% load docs
input_filename = "training_providers_merged.pkl"

with open(data_path / input_filename, "rb") as f:
    docs = load(f)

# %% set up a small embedding model
bge_model = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceBgeEmbeddings(model_name=bge_model)

# %% set up vector store location
output_dir = root_path / "data" / "vectorstores"
if not output_dir.exists():
    output_dir.mkdir()
vectorestore_file = output_dir / input_filename.split(".")[0] / bge_model

# %% set up a vector store
db = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory=str(vectorestore_file)
)
# %%
