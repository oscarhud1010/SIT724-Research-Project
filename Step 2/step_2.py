import json
import base64
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler
from datetime import datetime

# Model/Image/Question file paths
IMAGE_PATH = 'images/control_image_1.PNG'
MODEL_PATH = 'model/Qwen2.5-VL-72B-Instruct-q4_k_m.gguf'
CLIP_MODEL_PATH = 'model/Qwen2.5-VL-72B-Instruct-mmproj-f16.gguf'
QUESTIONS_FILE_PATH = 'step_2_questions.json'

# Choose how many simulations to run and the base seed to start from
NUM_SIMULATIONS = 20
base_seed = 1400

# RECORD KEEPING - Create output file, explain what this simulations aims to do, name file based on date run
# keep record of model, image, question file and any other relevant variables
SIMULATION_TYPE = 'Step 2: Show the LLM can understand and logically answer the UEQ questions - rate UI as bad, temperature 0.7 not 0.0'

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
    n_ctx=8192,  # Increased context for batch processing
    
    # Optimize
    verbose=False,  # Disable verbose output for speed
    n_threads=8,  # Increase threads 
    n_batch=512,  # Increase batch size
)

# ----------------- CREATE PROMPT -----------------
def create_batch_prompt(questions_list):
    
    # This is the main test for step 2 - Test if the LLM can understand the criteria properly
    # The exact same code was used each time, however swapping the word "WORST" in this prompt with the word "BEST" and the description to "great, high-quality"
    prompt = "For each bipolar scale below, assume you are rating the WORST POSSIBLE interface. Tell me which number (1 or 7) you would choose to indicate a terrible, low-quality interface.\n\n"

    for i, question in enumerate(questions_list, 1):
        prompt += f"Question {i}: {question}\n"

    prompt += f"\nRespond with a JSON object in this format. No markdown, no code blocks, no extra text, no extra white spaces, include quotation marks.\n"
    prompt +=f"{{\"q1\": 1, \"q2\": 7, ..., \"q{len(questions_list)}\": 7}}\n"

    
    return prompt

batch_prompt = create_batch_prompt(questions_list)

# Check prompt looks correct
print(batch_prompt)

# ----------------- IMAGE VERIFICATION -------------
'''
print("Testing image understanding")
description_response = llm.create_chat_completion(
    messages=[
        {
            "role": "system", 
            "content": "You are a helpful assistant. Describe what you see in the image in a few sentences."
            #"content": "You are a helpful assistant. Respond only with the requested JSON format when answering questions. Do not change the formatting or add any additional marks or punctuation"
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

'''
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
                "content": "You are a helpful assistant who does what the prompt says"
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

