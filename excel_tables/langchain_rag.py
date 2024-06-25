import warnings
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings('ignore')
_ = load_dotenv()

loader = UnstructuredExcelLoader("../docs/Complex_Order_Customer_Details.xlsx",
                                 mode="elements",
                                 infer_table_structure=True,)
docs = loader.load()

# export the documents to a markdown file
for doc in docs:
    with open(f"../docs/{doc.metadata['page_name']}.md", "w") as f:
        content_metadata = doc.page_content + "\n\n" + str(doc.metadata)
        f.write(content_metadata)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                               chunk_overlap=200)

splits = text_splitter.split_documents(docs)

documents = filter_complex_metadata(splits)

vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model_name="gpt-4o",
                 temperature=0)

prompt = hub.pull("rlm/rag-prompt")


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# Question
response = rag_chain.invoke("What is the total revenue in Q1 FY25?")
print(response)
