import os
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from textwrap import fill
import time


# Set your OpenAI API key (replace 'your-api-key' with your actual API key)
# os.environ["OPENAI_API_KEY"] = "your-api-key"

# Function to load and split the PDF into chunks
def load_pdf(file_path):
    
    print(f"Loading PDF ...")
    start = time.time()

    loader = PyPDFLoader(file_path)
    # Load the PDF content
    documents = loader.load()
    
    # Split documents into manageable chunks for processing
    docs = split_doc_by_char(documents, 8000, 100, 21)
    
    end = time.time()
    print(f"Loaded PDF in {(end-start):.2f} seconds")

    return docs

def split_doc_by_char(documents, chunkSize, chunkOverlap, endPage):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkOverlap)
    split_text = text_splitter.split_documents(documents)
    text_splitter_no_last_page = [x for x in split_text if x.metadata['page'] < endPage]
    return text_splitter_no_last_page

def define_llm():
    return ChatOpenAI(
        temperature=0.1,
        model_name="tinyllama:latest",
        api_key="ollama",
        #base_url="http://192.168.1.137:11434/v1",
        base_url="http://localhost:11434/v1",
    )

# Summarization function that chains map and reduce methods
def map_reduce_summarization(docs):
    
    llm = define_llm()
    
    # Step 1: Use the 'map' approach to create a summary for each chunk
    print(f"Mapping ...")
    start = time.time()
    map_chain = load_summarize_chain(llm, chain_type="map_reduce")
    intermediate_summaries = map_chain.invoke(docs)
    end = time.time()
    print(f"Mapped in {(end-start):.2f} seconds")


    # Step 2: Use the 'reduce' approach to further summarize the intermediate summaries
    print(f"Reducing ...")
    start = time.time()
    reduce_chain = load_summarize_chain(llm, chain_type="map_reduce")
    final_summary = reduce_chain.invoke(intermediate_summaries)
    end = time.time()
    print(f"Reduced in {(end-start):.2f} seconds")

    return final_summary

# Summarization function that chains stuff method
def stuff_summarization(docs):
    
    llm = define_llm()
    
    print(f"Stuffing ...")
    start = time.time()
    map_chain = load_summarize_chain(llm, chain_type="stuff")
    final_summary = map_chain.invoke(docs)
    end = time.time()
    print(f"Stuffed in {(end-start):.2f} seconds")

    return final_summary

# Main function to run the summarization
def main(pdf_file):
    print(f"Summarizing ...")
    start = time.time()
    # Load the PDF file
    docs = load_pdf(pdf_file)
    
    # Apply the stuff summarization
    #final_summary = stuff_summarization(docs)

    # Apply the map/reduce summarization
    final_summary = map_reduce_summarization(docs)

    end = time.time()
    print(f"Summarized in {(end-start):.2f} seconds")

    # Output the final summary
    with open("output.txt", "w", errors="ignore") as f:
        f.write(fill(final_summary["output_text"]))    

    print("\n\n=== Final Summary ===")
    print(fill(final_summary["output_text"]))

if __name__ == "__main__":
    # Replace 'your-pdf-file.pdf' with your actual PDF file path
    pdf_file_path = 'document.pdf'
    main(pdf_file_path)
