import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# u ovom modulu će se prikazati na koji način se dobivene informacije mogu strukturirati na smisleni način
# u primjeru je nekoliko cv-a iz kojih se izvlače najbitnije informacije i postavljaju unutar DataFrame-a

# učitaj dokumente
from pathlib import Path
from llama_index import download_loader
import re

PyMuPDFReader = download_loader("PyMuPDFReader")
loader = PyMuPDFReader()

pdf_files = Path("./resumes/").glob("*.pdf")

# učitaj OpenAIPydanticProgram, koristi se isključivo gpt-4 jer stariji modeli nisu dovoljno dobri za ovakve zadatke
from llama_index.program import (
    OpenAIPydanticProgram,
    DataFrameRowsOnly,
)
from llama_index.llms import OpenAI

program = OpenAIPydanticProgram.from_defaults(
    output_cls=DataFrameRowsOnly,
    llm=OpenAI(temperature=0, model="gpt-4-1106-preview"), # gpt-4-trubo
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

# analiziraj dokumente
rows_list =[]
for i, pdf_file in enumerate(pdf_files):
    document = loader.load_data(file_path=str(pdf_file), metadata=True)
    doc_text = ""
    file = ""

    for doc in document:
        # ako tekst sadrži više od jednog novog reda, postavi samo jedan novi red
        file = os.path.basename(doc.metadata['file_path'])
        doc_text = doc_text + re.sub(r'\n\s*\n', '\n', doc.text)

    # if i == 0: # limitacija za analiziranje samo jednog dokumenta
    res = program(
        input_str=doc_text
    )
    row = res.rows[0].row_values
    row.append(file)
    rows_list.append(row)

# strukturiranje rezultata u tablicu
import pandas as pd

columns = ['Name', 'Birth date', 'City', 'Current company', 'Profession', 'Year of experience', 'Technologies', 'E-mail', 'Phone', 'File']
df = pd.DataFrame(rows_list, columns=columns)
print(df)
