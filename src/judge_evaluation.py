import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

OUTPUT_JSONL_PATH = os.path.join(PROJECT_ROOT, "results", "judge_evaluations.jsonl")

import json
import requests
import base64
import time
import signal
from PIL import Image
from io import BytesIO

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-5-mini-2025-08-07")
api_version = os.getenv("OPENAI_API_VERSION", "2025-04-01-preview")

with open(os.path.join(PROJECT_ROOT, "credentials", "gpt_5_mini_api_key.json"), "r") as f:
    api_key_data = json.load(f)
    api_key = api_key_data["api_key"]

with open(os.path.join(PROJECT_ROOT, "prompts", "v3_judge_offline_all_factors.txt"), "r", encoding='utf-8') as f:
    offline_judge_prompt = f.read()

with open(os.path.join(PROJECT_ROOT, "prompts", "v3_judge_online_all_factors.txt"), "r", encoding='utf-8') as f:
    online_judge_prompt = f.read()

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Processing timeout exceeded")

def encode_image_to_base64(image_path, max_size=(512, 512), quality=85):
    try:
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            buffer.seek(0)
            
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

def get_instruction_from_file(instruction_path):
    with open(instruction_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def test_model_capabilities():
    print("Testing model capabilities...")
    
    test_messages = [
        {
            "role": "user",
            "content": "Hello, can you respond with 'Model is working'?"
        }
    ]
    
    base_path = f'deployments/{deployment}/chat/completions'
    params = f'?api-version={api_version}'
    test_url = f"{endpoint}/{base_path}{params}"
    
    try:
        response = requests.post(
            test_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "messages": test_messages,
                "max_completion_tokens": 50
            }
        )
        
        if response.status_code == 200:
            response_data = response.json()
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                print(f"Model responds to text: {content}")
                return True
            else:
                print(f"Model response structure issue: {response_data}")
                return False
        else:
            print(f"Model test failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Model test error: {e}")
        return False

def call_azure_openai_with_images(image1_path, image2_path, instruction, base_name, prompt_template, image1_placeholder, image2_placeholder, max_retries=3):
    
    image1_b64 = encode_image_to_base64(image1_path)
    image2_b64 = encode_image_to_base64(image2_path)
    
    base_path = f'deployments/{deployment}/chat/completions'
    params = f'?api-version={api_version}'
    generation_url = f"{endpoint}/{base_path}{params}"
    
    complete_prompt = prompt_template.replace("[text instruction]", instruction)
    complete_prompt = complete_prompt.replace(image1_placeholder, "")
    complete_prompt = complete_prompt.replace(image2_placeholder, "")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": complete_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image1_b64}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image2_b64}"
                    }
                }
            ]
        }
    ]
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                generation_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "messages": messages,
                    "max_completion_tokens": 5096
                }
            )
            
            if response.status_code == 200:
                response_data = response.json()
                print(f"API Response Status: {response.status_code}")
                print(f"Response Data Keys: {list(response_data.keys())}")
                
                if "choices" not in response_data:
                    print(f"No 'choices' in response: {response_data}")
                    return None
                
                if len(response_data["choices"]) == 0:
                    print(f"Empty choices array: {response_data}")
                    return None
                
                choice = response_data["choices"][0]
                print(f"Choice Keys: {list(choice.keys())}")
                
                if "message" not in choice:
                    print(f"No 'message' in choice: {choice}")
                    return None
                
                if "content" not in choice["message"]:
                    print(f"No 'content' in message: {choice['message']}")
                    return None
                
                model_output = choice["message"]["content"]
                
                if not model_output or model_output.strip() == "":
                    print(f"Empty model output!")
                    print(f"Full response data: {response_data}")
                    return None
                
                print("\n" + "="*80)
                print("MODEL OUTPUT:")
                print("="*80)
                print(model_output)
                print("="*80 + "\n")
                
                return model_output
            elif response.status_code == 429:
                wait_time = 60 * (attempt + 1)
                print(f"Rate limit exceeded. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
                continue
            else:
                print(f"Error calling Azure OpenAI: {response.status_code} - {response.text}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(5)
                
        except Exception as e:
            print(f"Exception calling Azure OpenAI: {str(e)}")
            if attempt == max_retries - 1:
                return None
            time.sleep(5)
    
    return None

def extract_json_from_response(response_text):
    try:
        print("\n" + "-"*60)
        print("JSON EXTRACTION PROCESS:")
        print("-"*60)
        print(f"Raw response length: {len(response_text)} characters")
        print(f"Raw response preview: {response_text[:300]}...")
        
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            print(f"Found JSON block from position {start_idx} to {end_idx}")
            print(f"Extracted JSON length: {len(json_str)} characters")
            print(f"Extracted JSON preview: {json_str[:200]}...")
            
            try:
                parsed_json = json.loads(json_str)
                print("JSON parsing successful!")
                print("-"*60 + "\n")
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                json_str = json_str.replace('\n', ' ').replace('\r', ' ')
                print("Trying to fix JSON by removing newlines...")
                try:
                    parsed_json = json.loads(json_str)
                    print("JSON parsing successful after fix!")
                    print("-"*60 + "\n")
                    return parsed_json
                except json.JSONDecodeError as e2:
                    print(f"Still failed after fix: {e2}")
                    print(f"Problematic JSON: {json_str[:500]}...")
                    print("-"*60 + "\n")
                    return None
        
        if '```json' in response_text:
            print("Trying Method 2: Looking for JSON in ```json code blocks...")
            start_marker = response_text.find('```json') + 7
            end_marker = response_text.find('```', start_marker)
            if end_marker != -1:
                json_str = response_text[start_marker:end_marker].strip()
                print(f"Found JSON in ```json code block")
                print(f"JSON length: {len(json_str)} characters")
                print(f"JSON preview: {json_str[:200]}...")
                try:
                    parsed_json = json.loads(json_str)
                    print("JSON parsing successful from code block!")
                    print("-"*60 + "\n")
                    return parsed_json
                except json.JSONDecodeError as e:
                    print(f"JSON decode error from code block: {e}")
        
        if '```' in response_text:
            print("Trying Method 3: Looking for JSON in ``` code blocks...")
            start_marker = response_text.find('```') + 3
            end_marker = response_text.find('```', start_marker)
            if end_marker != -1:
                json_str = response_text[start_marker:end_marker].strip()
                if json_str.startswith('{'):
                    print(f"Found JSON in ``` code block")
                    print(f"JSON length: {len(json_str)} characters")
                    print(f"JSON preview: {json_str[:200]}...")
                    try:
                        parsed_json = json.loads(json_str)
                        print("JSON parsing successful from code block!")
                        print("-"*60 + "\n")
                        return parsed_json
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error from code block: {e}")
        
        print("No JSON found in response")
        print(f"Full response: {response_text}")
        print("-"*60 + "\n")
        return None
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"JSON string that failed: {json_str}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def get_all_api_images():
    api_img_dir = os.path.join(PROJECT_ROOT, "HumanEdit", "api_img_400")
    return [f for f in os.listdir(api_img_dir) if f.endswith('.png')]

def process_single_evaluation(api_image_name, timeout_seconds=300):
    base_name = api_image_name.replace('.png', '')
    
    ground_truth_path = os.path.join(PROJECT_ROOT, "HumanEdit", "gt_img_400", api_image_name)
    edited_path = os.path.join(PROJECT_ROOT, "HumanEdit", "api_img_400", api_image_name)
    input_path = os.path.join(PROJECT_ROOT, "HumanEdit", "input_img_400", api_image_name)
    instruction_path = os.path.join(PROJECT_ROOT, "HumanEdit", "instructions_400", f"{base_name}.txt")
    
    required_files = [ground_truth_path, edited_path, input_path, instruction_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            return None
    
    instruction = get_instruction_from_file(instruction_path)
    
    print(f"Processing evaluation for: {base_name}")
    print(f"Instruction: {instruction}")
    print(f"Timeout set to: {timeout_seconds} seconds ({timeout_seconds/60:.1f} minutes)")
    print("\n" + "="*80)
    print(f"EVALUATING: {base_name}")
    print("="*80)
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        print("Processing OFFLINE setting...")
        offline_response = call_azure_openai_with_images(
            ground_truth_path, 
            edited_path, 
            instruction,
            base_name,
            offline_judge_prompt,
            "[ground truth image]",
            "[edited image]"
        )
        
        if offline_response is None:
            print(f"Failed to get offline response for {base_name}")
            print("="*80 + "\n")
            return None
        
        print(f"Extracting JSON from offline response for {base_name}...")
        offline_result = extract_json_from_response(offline_response)
        
        if offline_result is None:
            print(f"Failed to parse offline JSON for {base_name}")
            print("="*80 + "\n")
            return None
        
        print(f"Successfully parsed offline JSON for {base_name}")
        
        print("Processing ONLINE setting...")
        online_response = call_azure_openai_with_images(
            input_path, 
            edited_path, 
            instruction,
            base_name,
            online_judge_prompt,
            "[input image]",
            "[edited image]"
        )
        
        if online_response is None:
            print(f"Failed to get online response for {base_name}")
            print("="*80 + "\n")
            return None
        
        print(f"Extracting JSON from online response for {base_name}...")
        online_result = extract_json_from_response(online_response)
        
        if online_result is None:
            print(f"Failed to parse online JSON for {base_name}")
            print("="*80 + "\n")
            return None
        
        print(f"Successfully parsed online JSON for {base_name}")
        
        merged_result = {
            "image_id": base_name,
            "offline_factor_results": offline_result.get("offline_factor_results", {}),
            "online_factor_results": online_result.get("online_factor_results", {})
        }
        
        print(f"Successfully merged results for {base_name}")
        print("="*80 + "\n")
        
        return merged_result
        
    except TimeoutError:
        print(f"TIMEOUT: Processing {base_name} exceeded {timeout_seconds} seconds ({timeout_seconds/60:.1f} minutes)")
        print(f"Skipping {base_name} due to timeout")
        print("="*80 + "\n")
        return None
    except Exception as e:
        print(f"Unexpected error processing {base_name}: {e}")
        print("="*80 + "\n")
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def save_to_jsonl(evaluations, output_path, mode='a'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, mode, encoding='utf-8') as f:
        for evaluation in evaluations:
            if evaluation is not None:
                f.write(json.dumps(evaluation, ensure_ascii=False) + '\n')
    
    if mode == 'a':
        print(f"Results appended to: {output_path}")
    else:
        print(f"Results saved to: {output_path}")

def main():
    print("Starting judge evaluation process for both OFFLINE and ONLINE settings...")
    
    if not test_model_capabilities():
        print("Model test failed. Please check your model configuration.")
        return
    
    api_images = get_all_api_images()
    print(f"Found {len(api_images)} API images to evaluate")
    
    processed_ids = set()
    if os.path.exists(OUTPUT_JSONL_PATH):
        with open(OUTPUT_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'image_id' in data:
                        processed_ids.add(data['image_id'])
                except:
                    continue
        print(f"Found {len(processed_ids)} already processed images")
    
    evaluations = []
    skipped = 0
    timeout_count = 0
    
    for i, api_image in enumerate(api_images):
        base_name = api_image.replace('.png', '')
        
        if base_name in processed_ids:
            print(f"\nSkipping {i+1}/{len(api_images)}: {api_image} (already processed)")
            skipped += 1
            continue
            
        print(f"\nProcessing {i+1}/{len(api_images)}: {api_image}")
        print("Processing both OFFLINE and ONLINE settings...")
        
        start_time = time.time()
        evaluation = process_single_evaluation(api_image)
        end_time = time.time()
        processing_time = end_time - start_time
        
        evaluations.append(evaluation)
        
        if evaluation is not None:
            print(f"Successfully evaluated {api_image} (took {processing_time:.1f}s)")
            print(f"Results include both offline and online factor results")
            save_to_jsonl([evaluation], OUTPUT_JSONL_PATH)
            print(f"Results saved to: {OUTPUT_JSONL_PATH}")
        else:
            if processing_time >= 290:
                timeout_count += 1
                print(f"Timeout: {api_image} (took {processing_time:.1f}s)")
            else:
                print(f"Failed to evaluate {api_image} (took {processing_time:.1f}s)")
        
        if i < len(api_images) - 1:
            print("Waiting 10 seconds before next request...")
            time.sleep(10)
    
    successful = sum(1 for e in evaluations if e is not None)
    failed = len(evaluations) - successful
    print(f"\n=== Evaluation Complete ===")
    print(f"Total images found: {len(api_images)}")
    print(f"Already processed: {skipped}")
    print(f"Newly processed: {len(evaluations)}")
    print(f"Successfully evaluated: {successful}")
    print(f"Failed (including timeouts): {failed}")
    print(f"Timeouts: {timeout_count}")
    print(f"Results saved to: {OUTPUT_JSONL_PATH}")
    print("Each result includes both offline_factor_results and online_factor_results")

if __name__ == "__main__":
    main()
