import warnings
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.storage import InMemoryStore

warnings.filterwarnings('ignore')
_ = load_dotenv()

input_html = "../docs/nvidia_financial_results_q1_fiscal_2025.html"

# Parse the HTML content
loader = UnstructuredHTMLLoader(input_html)

docs = loader.load()

print("there are ", len(docs), " documents in the loader")

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4096)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=1024)

vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=OpenAIEmbeddings()
)
# The storage layer for the parent documents
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(docs, ids=None)

print("there are ", len(list(store.yield_keys())), " keys in the store")

retrieved_docs = retriever.invoke("$2,210")

for doc in retrieved_docs:
    print(doc.page_content)
    print('\n')
    print(doc.metadata)




