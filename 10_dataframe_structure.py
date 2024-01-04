import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# u ovom modulu će se prikazati na koji način se mogu strukturirati informacije dobivene iz dokumenata
# u primjeru je nekoliko cv-a iz kojih se izvlače najbitnije informacije te se na temelju toga
# dolazi do zaključka koji kandidati su najprikladniji za developer poziciju

from llama_index.program import (
    OpenAIPydanticProgram,
    DFFullProgram,
    DFRowsProgram,
)
import pandas as pd
from llama_index.llms import OpenAI

from llama_index.program import (
    OpenAIPydanticProgram,
    DFFullProgram,
    DataFrame,
    DataFrameRowsOnly,
)
from llama_index.llms import OpenAI

# inicijaliziraj prazan dataframe
df = pd.DataFrame(
    {
        "Name": pd.Series(dtype="str"),
        "Birth year": pd.Series(dtype="int"),
        "City": pd.Series(dtype="str"),
        "Proffesion": pd.Series(dtype="str"),
        "Work experience": pd.Series(dtype="str"),
        "Technologies": pd.Series(dtype="str"),
    }
)

# initialize program, using existing df as schema
df_rows_program = DFRowsProgram.from_defaults(
    pydantic_program_cls=OpenAIPydanticProgram, 
    df=df
)

# učitaj dokumente
from pathlib import Path
from llama_index import download_loader
import re

PyMuPDFReader = download_loader("PyMuPDFReader")
loader = PyMuPDFReader()

pdf_files = Path("./resumes/").glob("*.pdf")

program = OpenAIPydanticProgram.from_defaults(
    output_cls=DataFrameRowsOnly,
    llm=OpenAI(temperature=0, model="gpt-4-1106-preview"),
    prompt_template_str=(
        "Please extract the following text into a structured data:"
        " {input_str}. The column names are the following: ['Name', 'Birth date',"
        " 'City', 'Current company', 'Proffesion', 'Years of experience', 'Technologies', 'E-mail', 'Phone']. "
        " For tecnologies column, extract only programming languages, if there are none set value to null. For proffesion column, write the industry in which the person is operating. "
        " Do not specify additional parameters that"
        " are not in the function schema. "
    ),
    verbose=True,
)

rows =[]
for i, pdf_file in enumerate(pdf_files):
    document = loader.load_data(file_path=str(pdf_file), metadata=True)
    doc_text = ""
    for doc in document:
        # ako tekst sadrži više od jednog novog reda, postavi samo jedan novi red
        doc_text = doc_text + re.sub(r'\n\s*\n', '\n', doc.text)

    res = program(
        input_str=doc_text
    )
    rows.append(res.rows[0].row_values)

columns = ['Name', 'Birth date', 'City', 'Current company', 'Profession', 'Year of experience', 'Technologies', 'E-mail', 'Phone']
df = pd.DataFrame(rows, columns=columns)
print(df)