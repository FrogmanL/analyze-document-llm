import os
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
import time

# Set your OpenAI API key (replace 'your-api-key' with your actual API key)
# os.environ["OPENAI_API_KEY"] = "your-api-key"

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
