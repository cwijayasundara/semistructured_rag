Install the required packages by running the following command in the terminal:

```bash
pip install -r requirements.txt
```

You need do create a .env with the below keys:

OPENAI_API_KEY=''
LANGCHAIN_API_KEY=''

The PDF partitioning used by Unstructured will use:

tesseract for Optical Character Recognition (OCR)
poppler for PDF rendering and processing

```bash
! brew install tesseract
! brew install poppler
```

How to run the application:

```bash
cd html_tables
python app.py
```