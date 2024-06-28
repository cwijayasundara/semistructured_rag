from bs4 import BeautifulSoup
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import warnings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings('ignore')
_ = load_dotenv()

# Load HTML content from a file
file_path = '../docs/nvidia_financial_results_q1_fiscal_2025.html'

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=200)

with open(file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

# Parse the HTML content
soup = BeautifulSoup(html_content, 'html.parser')

# Extract plain text (ignoring script and style content)
for script_or_style in soup(['script', 'style']):
    script_or_style.extract()

plain_text = soup.get_text(separator=' ', strip=True)

# Extract headers
headers = {}
for level in range(1, 7):  # Loop through h1 to h6
    header_tag = f'h{level}'
    headers[header_tag] = [header.get_text(strip=True) for header in soup.find_all(header_tag)]

# Extract tables and their rows
tables = []
for table in soup.find_all('table'):
    table_data = []
    rows = table.find_all('tr')
    for row in rows:
        cells = row.find_all(['td', 'th'])  # 'td' for data cells, 'th' for header cells
        cell_texts = [cell.get_text(strip=True) for cell in cells]
        table_data.append(cell_texts)
    tables.append(table_data)

text_to_embed = [plain_text]

for tag, content in headers.items():
    text_to_embed.append(content)

for idx, table in enumerate(tables):
    for row in table:
        # remove tab characters and new lines characters from the row
        row = [cell.replace('\t', '').replace('\n', '') for cell in row]
        # create a string from the row
        row = ' '.join(row)
        # if the row is large then split it into smaller chunks to avoid hitting the OpenAI API limit
        splits = text_splitter.split_text(row)
        text_to_embed.append(splits)

documents = []
for text in text_to_embed:
    if isinstance(text, list):
        text = ' '.join(text)
    # check if the text is not empty or not None
    if text and text != 'None':
        documents.append(Document(page_content=text))

# loop through the documents and remove any None values
for document in documents:
    if document.page_content is None:
        documents.remove(document)

embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(documents, embeddings)

query = "$2,501"

retriever = vectorstore.as_retriever()
result = retriever.invoke(query, k=4)

for i in range(len(result)):
    print(result[i])
