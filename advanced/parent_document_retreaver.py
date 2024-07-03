import warnings
from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings('ignore')
_ = load_dotenv()

loaders = [
    WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/"),
    WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# This text splitter is used to create the parent documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
# This text splitter is used to create the child documents
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
# The vectorstore to use to index the child chunks
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

# call the vectors store to get the child documents
sub_docs = vectorstore.similarity_search("Memory in agents")

# for doc in sub_docs:
#     print(doc)

# call the retriever to get the parent document
retrieved_docs = retriever.invoke("Memory in agents")

for doc in retrieved_docs:
    print(doc.page_content)



