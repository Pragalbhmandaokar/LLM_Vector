# Arrow/NICD Train Air

Prototyping semantic search for training provider and trainers dataset

## Code: Getting started

- Set up a Python virtual environment and install required packages from [requirements.txt](./requirements.txt);
- `demo_bm25_retriever.py` implements a semantic search system by combining BM25 retrieval with HuggingFace embeddings. Requires `load_csv_2_docs.py` for document preprocessing and `embed_docs.py` for document embedding to be run prior.
- `demo_semantic_search_retriever.py` implements a semantic search system using HuggingFace embeddings and Chroma vector stores. It demonstrates two retrieval methods: similarity-based search and Maximal Marginal Relevance (MMR) search, using a pre-trained embedding model (`bge-small-en-v1.5`) to convert text into vector embeddings that capture the semantic meaning of the documents. Requires `load_csv_2_docs.py` for document preprocessing and `embed_docs.py` for document embedding to be run prior.
- `demo_openai.py` implements a semantic search system using OpenAI embeddings and Chroma vector database. Requires `load_csv_2_docs.py` for document preprocessing and `embed_docs.py` for document embedding to be run prior.
- `trainers_preprocess.py` - This code loads trainer details from an Excel file into a pandas DataFrame and then converts the DataFrame into a format suitable for use with LangChain. It sets up an embedding model using HuggingFace’s BGE and creates a vector store using Chroma, storing the embeddings and documents in a specified directory. This setup facilitates efficient document retrieval and similarity search based on the trainer descriptions.
- `trainers_semantic_search.py` - This script loads the previously created vector store and executes a single query using similarity search. The search parameters `search_kwargs` can be modified to perform a filter on a metadata column before the semantic search is executed. This script is useful for experimentation.
- `trainers_semantic_search_with_eval.py` - This script is an updated version of the \path{trainers_semantic_search.py} with the main difference that this script loads all test queries and executes the semantic search for each of them with the corresponding expected results.


This prototype code was developed by National Innovation Centre for Data, Newcastle University as part of the Arrow Innovation Support programme. If you have any queries following the Arrow project, please contact NICD Lead for Arrow programme Peter Michalak (Peter.Michalak@newcastle.ac.uk).