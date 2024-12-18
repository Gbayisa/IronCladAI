# llm_server_cpu_smaller.py

from flask import Flask, request, jsonify
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import time
import psutil  # For memory usage tracking
from tqdm import tqdm  # For progress indication

# Set environment variables for CPU optimization
os.environ["OMP_NUM_THREADS"] = "4"  # Set the number of OpenMP threads to use
os.environ["MKL_NUM_THREADS"] = "4"  # Set the number of threads for Intel's Math Kernel Library
torch.set_num_threads(4)  # Control the number of CPU threads
torch.set_num_interop_threads(4)  # Control the number of inter-op threads

# Define the model name for the smaller LLaMA-2-3B model
model_name = "meta-llama/Llama-2-7b-hf"  # Use the 3B model instead of the 7B model

print(f"Loading {model_name} model and tokenizer with CPU-only optimization...")

# Load the tokenizer without any specific configuration for quantization
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the LLaMA-2-3B model with CPU-only settings
try:
    # Load model with reduced memory usage configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,  # Sequential loading for reduced memory
        torch_dtype=torch.float16,  # Use float16 precision to reduce memory usage
        device_map=None,  # Make sure no GPU-specific configurations are used
    )
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

print(f"Model and tokenizer loaded successfully for {model_name} with CPU-friendly configurations.")

# Set up the text generation pipeline for CPU usage
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Create the Flask app
app = Flask(__name__)

# Global variables to store the data
df = None
text_content = None

# Define a function to check memory usage
def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / 1024 ** 2:.2f} MB")

@app.route('/')
def home():
    return "LLM Data Analysis Server is Running!"

# Endpoint to upload a file (Excel or Word)
@app.route('/upload', methods=['POST'])
def upload_file():
    global df, text_content
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    filename = file.filename

    try:
        if filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
            text_content = None
            print(f"Excel file uploaded successfully: {filename}")
            return jsonify({"message": "Excel file uploaded successfully", "columns": list(df.columns)})
        elif filename.endswith('.docx'):
            import docx
            doc = docx.Document(file)
            text_content = "\n".join([para.text for para in doc.paragraphs])
            df = None
            print(f"Word document uploaded successfully: {filename}")
            return jsonify({"message": "Word document uploaded successfully"})
        else:
            return jsonify({"error": "Unsupported file format"}), 400
    except Exception as e:
        print(f"File upload failed: {str(e)}")
        return jsonify({"error": f"File upload failed: {str(e)}"}), 500

# Endpoint to analyze the data based on natural language query
@app.route('/analyze', methods=['POST'])
def analyze():
    global df, text_content

    # Get the query from the request
    query = request.json['query']
    print(f"Received query: {query}")

    # Check if any file content (Excel or Word) is uploaded
    if df is None and text_content is None:
        prompt = f"{query}"
        print(f"Generated prompt for general query: {prompt}")
    else:
        if df is not None:
            data_str = df.head(5).to_csv(index=False)  # Use only the first 5 rows as an example
            prompt = f"Given the following data in CSV format:\n{data_str}\n\nAnswer the following question:\n{query}"
            print(f"Generated prompt with Excel data: {prompt[:1000]}")  # Log only the first 1000 characters
        else:
            truncated_text = text_content[:1000]  # Use only the first 1000 characters of text
            prompt = f"Given the following document text:\n{truncated_text}\n\nAnswer the following question:\n{query}"
            print(f"Generated prompt with Word document: {prompt[:1000]}")  # Log only the first 1000 characters

    # Start timer
    start_time = time.time()
    print_memory_usage()  # Print initial memory usage

    try:
        # Show progress bar (simulation, actual generation may not reflect it)
        for _ in tqdm(range(100), desc="Processing Query", ncols=75):
            time.sleep(0.02)  # Simulating a progress bar

        # Generate the response
        response = generator(
            prompt,
            max_new_tokens=50,          # Limit the number of tokens generated
            temperature=0.2,            # Lower temperature for less randomness
            top_p=0.8,                  # Lower top_p for higher precision in responses
            do_sample=True,             # Enable sampling
        )[0]["generated_text"]

        # Stop timer
        end_time = time.time()
        total_time = end_time - start_time

        # Print memory usage after generation
        print_memory_usage()
        
        # Calculate and print the elapsed time
        print(f"Time taken for query: {total_time:.2f} seconds")

        answer = response[len(prompt):].strip()
        return jsonify({"response": answer})
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        return jsonify({"error": f"Failed to generate response: {str(e)}"}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
