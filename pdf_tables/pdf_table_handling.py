from dotenv import load_dotenv
import warnings
from langchain_community.vectorstores.utils import filter_complex_metadata
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements
from Utils import Utils
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

warnings.filterwarnings('ignore')

_ = load_dotenv()

utils = Utils()

DLAI_API_KEY = utils.get_dlai_api_key()
DLAI_API_URL = utils.get_dlai_url()

s = UnstructuredClient(
    api_key_auth=DLAI_API_KEY,
    server_url=DLAI_API_URL,
)

filename = "../docs/NVIDIAAn.pdf"

with open(filename, "rb") as f:
    files=shared.Files(
        content=f.read(),
        file_name=filename,
    )

req = shared.PartitionParameters(
    files=files,
    strategy="fast",
    hi_res_model_name="yolox",
    pdf_infer_table_structure=True,
    skip_infer_table_types=[],
)

try:
    resp = s.general.partition(req)
    pdf_elements = dict_to_elements(resp.elements)
except SDKError as e:
    print(e)

documents = []
for element in pdf_elements:
    metadata = element.metadata.to_dict()
    del metadata["languages"]
    metadata["source"] = metadata["filename"]
    documents.append(Document(page_content=element.text, metadata=metadata))

# Filter out elements with complex metadata that are not useful for the vector store
documents = filter_complex_metadata(documents)

embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(documents, embeddings)

query = "Whats the revenue for Q1 FY25?"

retriever = vectorstore.as_retriever()

result = retriever.invoke(query, k=3)

print("response from the retriever", result)

template = """Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(temperature=0,
                   model="gpt-4o")

# RAG pipeline
chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
)

query = "Whats the revenue for Q1 FY25?"

response = chain.invoke(query)
print("response from the chain", response)