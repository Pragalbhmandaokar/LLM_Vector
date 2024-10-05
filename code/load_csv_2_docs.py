# %% module imports
from janitor import clean_names
from langchain_community.document_loaders import DataFrameLoader
from pandas import read_excel
from pathlib import Path
from pickle import dump

# %% data paths
root_path = Path(__file__).resolve().parents[1]
data_path = root_path / "data" / "raw"

# %% load data from csv
input_filename = "training_providers_merged.xlsm"

df = read_excel(data_path / input_filename).clean_names()

# %% wrap df inside langchain docs
page_content_column = "description"
loader = DataFrameLoader(df, page_content_column)
docs = loader.load()

# %% pickle docs
output_dir = root_path / "data" / "pickles"
if not output_dir.exists():
    output_dir.mkdir()

output_file = output_dir / input_filename.replace(".xlsm", ".pkl")
with open(output_file, "wb") as f:
    dump(docs, f)


# %%
