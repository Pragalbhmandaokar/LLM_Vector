# %%
import os
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DataFrameLoader

# %% 
DATA_PATH = "./../data/Trainers details.xlsx"
VECTORSTORE_PATH = "./../data/vectorstores"

# %% load data from csv
df = pd.read_excel(DATA_PATH)

# %% wrap df inside langchain docs
page_content_column = "trainer_description"
loader = DataFrameLoader(df, page_content_column)
docs = loader.load()

# %% set up a small embedding model
bge_model = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceBgeEmbeddings(model_name=bge_model)

# %% set up vector store location
vectorestore_file = os.path.join(VECTORSTORE_PATH, DATA_PATH.split(".")[0], bge_model)

# %% set up a vector store
db = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory=str(vectorestore_file)
)