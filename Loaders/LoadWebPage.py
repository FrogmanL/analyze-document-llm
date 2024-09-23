from langchain_community.document_loaders import WebBaseLoader
import time
import Loaders.TextParser as TextParser

# Function to load and split the text into chunks
def load_web_page(url):
    
    print(f"Loading Web Page ...")
    start = time.time()

    loader = WebBaseLoader(url)
    # Load the PDF content
    documents = loader.load()
    
    # Split documents into manageable chunks for processing
    #It seems that higher chunk size is processed faster,
    #but lower chunk size produces better granularity is analysis results.
    docs = TextParser.split_doc_by_char(documents, 4000, 100)
    
    end = time.time()
    print(f"Loaded Text in {(end-start):.2f} seconds")

    return docs