import json
import base64
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler
from datetime import datetime

# Model/Image/Question file paths
IMAGE_PATH = 'images/control_image_1.PNG'
MODEL_PATH = 'model/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf'
CLIP_MODEL_PATH = 'model/mmproj-F16.gguf'
#QUESTIONS_FILE_PATH = 'step_2_questions.json'

# Choose how many simulations to run and the base seed to start from
NUM_SIMULATIONS = 10
base_seed = 1400

# RECORD KEEPING - Create output file, explain what this simulations aims to do, name file based on date run
# keep record of model, image, question file and any other relevant variables
SIMULATION_TYPE = 'Step 3.1: Show the LLM can "see" images properly --- Fine tune response format + run 10 simulations'

timestamp = datetime.now().strftime("%d-%m-%Y %I-%M%p")

# CHANGED TO .TXT FILE FOR STEP 3
output_filename = f"Answers/Answer {timestamp}.txt"

with open(output_filename, "w") as file:
    print(f"{SIMULATION_TYPE}\n", file=file)
    print(f"Running {NUM_SIMULATIONS} simulations with base seed: {base_seed}\n", file=file)
    print(f"Model {MODEL_PATH}\n", file=file)
    #print(f"Image: {IMAGE_PATH}\n", file=file)
    #print(f"Question file: {QUESTIONS_FILE_PATH}\n", file=file)

# Load question file
# with open(QUESTIONS_FILE_PATH, 'r') as f:
#    questions_list = json.load(f)

# Convert image into format LLM can see
def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

# -------- Step 3.1 Images ---------
image_paths = [
'images/image_1.PNG',
'images/image_2.png',
'images/image_3.png',
'images/image_4.png',
'images/image_5.png',
'images/image_6.png',
'images/image_7.png',
'images/image_8.png',
'images/image_9.png',
'images/image_10.PNG'
]

data_uris = [image_to_base64_data_uri(path) for path in image_paths]

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
# Prompt for step 3 is significantly shorter so simply write the prompt in the LLM construction code

# ---------------- SIMULATION -----------------------
for sim in range(NUM_SIMULATIONS):
    current_seed = base_seed + sim
    
    print(f'Simulation {sim + 1} (seed: {current_seed}): ', end='', flush=True)
    
    # Reset the LMM to use the new seed and clear internal state
    llm.reset()
    llm.set_seed(current_seed)

    # Write which simulation the image description is for - Need this because each simulation it not necessary on one line anymore
    with open(output_filename, "a") as file:
        print(f"Simulation {sim + 1}:", file=file)
    
    # Add new loop to show the model each image
    for image_id, data_uri in enumerate(data_uris):
        print(f'Image {image_id + 1}')

        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant who does what the prompt says"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the image."},
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=512,  
        )
    
        # Get response - doesnt have to be JSON for this step
        response_content = response['choices'][0]['message']['content']

        with open(output_filename, "a") as file:
                print(f"Image {image_id + 1}: {response_content}\n", file=file)
        
    with open(output_filename, "a") as file:
        print("\n" + "-" * 50 + "\n", file=file)
    
    print(f"Simulation {sim + 1} complete")

print('FINISHED')

