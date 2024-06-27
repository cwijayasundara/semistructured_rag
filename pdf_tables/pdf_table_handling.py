from dotenv import load_dotenv
import warnings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from unstructured.staging.base import dict_to_elements
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from langchain_openai import OpenAIEmbeddings
from Utils import Utils
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

warnings.filterwarnings('ignore')

_ = load_dotenv()

utils = Utils()

DLAI_API_KEY = utils.get_dlai_api_key()
DLAI_API_URL = utils.get_dlai_url()

s = UnstructuredClient(
    api_key_auth=DLAI_API_KEY,
    server_url=DLAI_API_URL,
)

# filename = "../docs/embedded-images-tables.pdf"
filename = "../docs/NVIDIAAn.pdf"

with open(filename, "rb") as f:
    files = shared.Files(
        content=f.read(),
        file_name=filename,
    )

req = shared.PartitionParameters(
    files=files,
    strategy="hi_res",
    chunking_strategy="by_title",
    hi_res_model_name="yolox",
    skip_infer_table_types=[],
    pdf_infer_table_structure=True,
)

try:
    resp = s.general.partition(req)
    elements = dict_to_elements(resp.elements)
except SDKError as e:
    print(e)


tables = [el for el in elements if el.category == "Table"]
print(len(tables))
table_html = tables[0].metadata.text_as_html
print(table_html)

# prep data for the Vector DB
documents = []
for element in elements:
    metadata = element.metadata.to_dict()
    del metadata["languages"]
    metadata["source"] = metadata["filename"]
    documents.append(Document(page_content=element.text, metadata=metadata))

embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(documents,
                                    embeddings)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

query = "Whats the basic net income per share for the three months ended April 28, 2024?"

result = retriever.invoke(query)

print("from the vector store", result)

template = """Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(temperature=0,
                   model="gpt-4o")

# RAG pipeline
chain = (
        {"context": retriever,
         "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
)

response = chain.invoke(query)
print("response from the chain", response)


