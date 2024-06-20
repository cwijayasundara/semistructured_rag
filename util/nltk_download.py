import ssl
import certifi
import nltk

# Apply certifi certificates
ssl._create_default_https_context = ssl._create_unverified_context

# Download the NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')