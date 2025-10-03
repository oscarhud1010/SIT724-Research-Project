import json
import base64
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler
from datetime import datetime

# Model/Image/Question file paths
IMAGE_PATH = 'images/control_image_1.PNG'
MODEL_PATH = 'model/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf'
CLIP_MODEL_PATH = 'model/mmproj-F16.gguf'
QUESTIONS_FILE_PATH = 'step_1_questions.json'

# Choose how many simulations to run and the base seed to start from
NUM_SIMULATIONS = 100
base_seed = 1400

# RECORD KEEPING - Create output file, explain what this simulations aims to do, name file based on date run
# keep record of model, image, question file and any other relevant variables
SIMULATION_TYPE = 'Step 1: Show the LLM can answer Likert Scale questions'

timestamp = datetime.now().strftime("%d-%m-%Y %I-%M%p")
output_filename = f"Answers/Answer {timestamp}.csv"

with open(output_filename, "w") as file:
    print(f"{SIMULATION_TYPE}\n", file=file)
    print(f"Running {NUM_SIMULATIONS} simulations with base seed: {base_seed}\n", file=file)
    print(f"Model {MODEL_PATH}\n", file=file)
    print(f"Image: {IMAGE_PATH}\n", file=file)
    print(f"Question file: {QUESTIONS_FILE_PATH}\n", file=file)

# Load question file
with open(QUESTIONS_FILE_PATH, 'r') as f:
    questions_list = json.load(f)

# Convert image into format LLM can see
def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

data_uri = image_to_base64_data_uri(IMAGE_PATH)

# Set up LLM with chosen parameters
chat_handler = Qwen25VLChatHandler(clip_model_path=CLIP_MODEL_PATH)

llm = Llama(
    # Model settings
    model_path=MODEL_PATH,
    chat_handler=chat_handler,
    chat_format="qwen2.5-vl",
    n_ctx=4096,  # Increased context for batch processing
    
    # Optimize
    verbose=False,  # Disable verbose output for speed
    n_threads=8,  # Increase threads 
    n_batch=512,  # Increase batch size
)

# ----------------- CREATE PROMPT -----------------
def create_batch_prompt(questions_list):
    
    prompt = "Answer the following 7 point bipolar likert-scale questions. Answer each question with ONLY a single number from 1-7.\n\n"
    
    for i, question in enumerate(questions_list, 1):
        prompt += f"Question {i}: {question}\n"
    
    prompt += f"\nRespond with a JSON object in this format. No markdown, no code blocks, no extra text, no extra white spaces. Include all quotation marks shown.\n"
    prompt +=f"{{\"q1\": \"q1 answer\", \"q2\": \"q2 answer\", ..., \"q{len(questions_list)}\": \"q{len(questions_list)} answer\"}}"
    return prompt

batch_prompt = create_batch_prompt(questions_list)

# Check prompt looks correct
print(batch_prompt)

# ----------------- IMAGE VERIFICATION -------------
print("Testing image understanding")
description_response = llm.create_chat_completion(
    messages=[
        {
            "role": "system", 
            "content": "You are a helpful assistant. Describe what you see in the image in a few sentences."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please describe what you see in this image."},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]
        }
    ],
    temperature=0.7,
    max_tokens=256,
)

description = description_response['choices'][0]['message']['content']
print(f"Image description: {description}")

# Write image description to the output file
with open(output_filename, "a") as file:
    print(f"Image description test: {description}\n", file=file)

# ---------------- SIMULATION -----------------------
for sim in range(NUM_SIMULATIONS):
    current_seed = base_seed + sim
    
    print(f'Simulation {sim + 1} (seed: {current_seed}): ', end='', flush=True)
    
    # Reset the LMM to use the new seed and clear internal state
    llm.reset()
    llm.set_seed(current_seed)
    
    # Single API call for all questions
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful assistant. Respond only with the requested JSON format when answering questions. Do not change the formatting or add any additional marks or punctuation"
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": batch_prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]
            }
        ],
        temperature=0.7,
        max_tokens=2048,  
    )
    
    # If response is in valid JSON format
    response_content = response['choices'][0]['message']['content']

    try:
        response_json = json.loads(response_content)

        with open(output_filename, "a") as file:
            ratings = [str(response_json[f"q{i}"]) for i in range(1, len(questions_list) + 1)]
            print(",".join(ratings), file=file)

    except json.JSONDecodeError as e:
        print("JSON parsing error (LLM did not respond as instructed)")
        print(f"Response content: {response_content}")

        with open(output_filename, "a") as file:
            print(f"JSON ERROR: ,{response_content}", file=file)
    
    print(f"Simulation {sim + 1} complete")

print('FINISHED')

