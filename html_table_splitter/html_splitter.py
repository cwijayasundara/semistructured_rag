from bs4 import BeautifulSoup
from langchain_core.documents import Document

# Load HTML content from a file
file_path = 'nested_table.html'

with open(file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

# Parse the HTML content
soup = BeautifulSoup(html_content, 'html.parser')

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

text_to_embed = []

for tag, content in headers.items():
    text_to_embed.append(content)

for idx, table in enumerate(tables):
    for row in table:
        # remove tab characters and new lines characters from the row
        row = [cell.replace('\t', '').replace('\n', '') for cell in row]
        # if the row is large then split it into smaller chunks to avoid hitting the OpenAI API limit
        text_to_embed.append(row)

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

# print the content of the table
for document in documents:
    print(document.page_content)
