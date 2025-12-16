# Resolve project root (one directory up from this file's directory)
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level from src/ to pipeline/

# Fixed output path for JSONL results (relative to new PROJECT_ROOT)
OUTPUT_JSONL_PATH = os.path.join(PROJECT_ROOT, "results", "judge_evaluations_all_v3_gemini.jsonl")

import json
import base64
import time
import signal
from PIL import Image
from io import BytesIO
from google import genai
from google.genai import types

# Gemini setup
YOUR_PROJECT_ID = ''
YOUR_LOCATION = ''

client = genai.Client(
    vertexai=True, project=YOUR_PROJECT_ID, location=YOUR_LOCATION,
)

model = "gemini-2.5-pro"

# Load judge prompts (relative to new PROJECT_ROOT)
with open(os.path.join(PROJECT_ROOT, "prompts", "v3_judge_offline_all_factors.txt"), "r", encoding='utf-8') as f:
    offline_judge_prompt = f.read()

with open(os.path.join(PROJECT_ROOT, "prompts", "v3_judge_online_all_factors.txt"), "r", encoding='utf-8') as f:
    online_judge_prompt = f.read()

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Processing timeout exceeded")

def encode_image_to_bytes(image_path, max_size=(512, 512), quality=85):
    """Encode image to bytes with compression to reduce token usage"""
    try:
        # Open and resize image to reduce token consumption
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Resize image to reduce size
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Save to bytes with compression
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            buffer.seek(0)
            
            return buffer.getvalue()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        # Fallback to original method
        with open(image_path, "rb") as image_file:
            return image_file.read()

def get_instruction_from_file(instruction_path):
    """Read instruction from text file"""
    with open(instruction_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def test_model_capabilities():
    """Test if the model supports image processing"""
    print("🧪 Testing model capabilities...")
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                "Hello, can you respond with 'Model is working'?"
            ],
        )
        
        if response and response.text:
            print(f"✅ Model responds to text: {response.text}")
            return True
        else:
            print(f"❌ Model response structure issue: {response}")
            return False
    except Exception as e:
        print(f"❌ Model test error: {e}")
        return False

def call_gemini_with_images(image1_path, image2_path, instruction, base_name, prompt_template, image1_placeholder, image2_placeholder, max_retries=3):
    """Call Gemini API with the three required inputs and smart retry mechanism"""
    
    # Read images as bytes
    image1_bytes = encode_image_to_bytes(image1_path)
    image2_bytes = encode_image_to_bytes(image2_path)
    
    # Prepare the prompt
    # Replace all placeholders in the prompt
    complete_prompt = prompt_template.replace("[text instruction]", instruction)
    # Remove image placeholders since images are passed as separate Part objects
    complete_prompt = complete_prompt.replace(image1_placeholder, "")
    complete_prompt = complete_prompt.replace(image2_placeholder, "")
    
    # Prepare contents for Gemini API
    contents = [
        complete_prompt,
        types.Part.from_bytes(
            data=image1_bytes,
            mime_type="image/jpeg",
        ),
        types.Part.from_bytes(
            data=image2_bytes,
            mime_type="image/jpeg",
        ),
    ]
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
            )
            
            if response and response.text:
                model_output = response.text
                
                # Print the model output to terminal
                print("\n" + "="*80)
                print("MODEL OUTPUT:")
                print("="*80)
                print(model_output)
                print("="*80 + "\n")
                
                return model_output
            else:
                print(f"❌ Empty or invalid response from Gemini API")
                print(f"Response: {response}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(5)  # Wait 5 seconds before retry
                
        except Exception as e:
            error_str = str(e)
            print(f"Exception calling Gemini API: {error_str}")
            
            # Check for rate limiting (429 or similar)
            if "429" in error_str or "rate limit" in error_str.lower() or "quota" in error_str.lower():
                wait_time = 60 * (attempt + 1)  # Exponential backoff
                print(f"Rate limit exceeded. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
                continue
            
            if attempt == max_retries - 1:
                return None
            time.sleep(5)  # Wait 5 seconds before retry
    
    return None

def extract_json_from_response(response_text):
    """Extract JSON from the response text with improved logic"""
    try:
        print("\n" + "-"*60)
        print("JSON EXTRACTION PROCESS:")
        print("-"*60)
        print(f"Raw response length: {len(response_text)} characters")
        print(f"Raw response preview: {response_text[:300]}...")
        
        # Method 1: Try to find JSON block in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            print(f"Found JSON block from position {start_idx} to {end_idx}")
            print(f"Extracted JSON length: {len(json_str)} characters")
            print(f"Extracted JSON preview: {json_str[:200]}...")
            
            try:
                parsed_json = json.loads(json_str)
                print("✅ JSON parsing successful!")
                print("-"*60 + "\n")
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                # Try to fix common JSON issues
                json_str = json_str.replace('\n', ' ').replace('\r', ' ')
                print("Trying to fix JSON by removing newlines...")
                try:
                    parsed_json = json.loads(json_str)
                    print("✅ JSON parsing successful after fix!")
                    print("-"*60 + "\n")
                    return parsed_json
                except json.JSONDecodeError as e2:
                    print(f"❌ Still failed after fix: {e2}")
                    print(f"Problematic JSON: {json_str[:500]}...")
                    print("-"*60 + "\n")
                    return None
        
        # Method 2: Try to find JSON in code blocks
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
                    print("✅ JSON parsing successful from code block!")
                    print("-"*60 + "\n")
                    return parsed_json
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error from code block: {e}")
        
        # Method 3: Try to find JSON in code blocks without language specifier
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
                        print("✅ JSON parsing successful from code block!")
                        print("-"*60 + "\n")
                        return parsed_json
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON decode error from code block: {e}")
        
        print("❌ No JSON found in response")
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
    """Get all images from api_img_400 directory"""
    api_img_dir = os.path.join(PROJECT_ROOT, "HumanEdit", "api_img_400")
    return [f for f in os.listdir(api_img_dir) if f.endswith('.png')]

def process_single_evaluation(api_image_name, timeout_seconds=300):
    """Process a single image evaluation with timeout control for both offline and online settings"""
    base_name = api_image_name.replace('.png', '')
    
    # Define paths for all required files
    ground_truth_path = os.path.join(PROJECT_ROOT, "HumanEdit", "gt_img_400", api_image_name)
    edited_path = os.path.join(PROJECT_ROOT, "HumanEdit", "api_img_400", api_image_name)
    input_path = os.path.join(PROJECT_ROOT, "HumanEdit", "input_img_400", api_image_name)
    instruction_path = os.path.join(PROJECT_ROOT, "HumanEdit", "instructions_400", f"{base_name}.txt")
    
    # Check if all required files exist
    required_files = [ground_truth_path, edited_path, input_path, instruction_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            return None
    
    # Read instruction
    instruction = get_instruction_from_file(instruction_path)
    
    print(f"Processing evaluation for: {base_name}")
    print(f"Instruction: {instruction}")
    print(f"⏰ Timeout set to: {timeout_seconds} seconds ({timeout_seconds/60:.1f} minutes)")
    print("\n" + "="*80)
    print(f"EVALUATING: {base_name}")
    print("="*80)
    
    # Set up timeout signal
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        # Process offline setting
        print("🔄 Processing OFFLINE setting...")
        offline_response = call_gemini_with_images(
            ground_truth_path, 
            edited_path, 
            instruction,
            base_name,
            offline_judge_prompt,
            "[ground truth image]",
            "[edited image]"
        )
        
        if offline_response is None:
            print(f"❌ Failed to get offline response for {base_name}")
            print("="*80 + "\n")
            return None
        
        # Extract JSON from offline response
        print(f"🔍 Extracting JSON from offline response for {base_name}...")
        offline_result = extract_json_from_response(offline_response)
        
        if offline_result is None:
            print(f"❌ Failed to parse offline JSON for {base_name}")
            print("="*80 + "\n")
            return None
        
        print(f"✅ Successfully parsed offline JSON for {base_name}")
        
        # Process online setting
        print("🔄 Processing ONLINE setting...")
        online_response = call_gemini_with_images(
            input_path, 
            edited_path, 
            instruction,
            base_name,
            online_judge_prompt,
            "[input image]",
            "[edited image]"
        )
        
        if online_response is None:
            print(f"❌ Failed to get online response for {base_name}")
            print("="*80 + "\n")
            return None
        
        # Extract JSON from online response
        print(f"🔍 Extracting JSON from online response for {base_name}...")
        online_result = extract_json_from_response(online_response)
        
        if online_result is None:
            print(f"❌ Failed to parse online JSON for {base_name}")
            print("="*80 + "\n")
            return None
        
        print(f"✅ Successfully parsed online JSON for {base_name}")
        
        # Merge results
        merged_result = {
            "image_id": base_name,
            "offline_factor_results": offline_result.get("offline_factor_results", {}),
            "online_factor_results": online_result.get("online_factor_results", {})
        }
        
        print(f"✅ Successfully merged results for {base_name}")
        print("="*80 + "\n")
        
        return merged_result
        
    except TimeoutError:
        print(f"⏰ TIMEOUT: Processing {base_name} exceeded {timeout_seconds} seconds ({timeout_seconds/60:.1f} minutes)")
        print(f"⏭️  Skipping {base_name} due to timeout")
        print("="*80 + "\n")
        return None
    except Exception as e:
        print(f"❌ Unexpected error processing {base_name}: {e}")
        print("="*80 + "\n")
        return None
    finally:
        # Restore the original signal handler and cancel alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def save_to_jsonl(evaluations, output_path, mode='a'):
    """Save evaluations to JSONL format with append mode"""
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
    """Main function to process all evaluations with real-time saving for both offline and online settings"""
    print("Starting judge evaluation process for both OFFLINE and ONLINE settings...")
    
    # Test model capabilities first
    if not test_model_capabilities():
        print("❌ Model test failed. Please check your model configuration.")
        return
    
    # Get all API images
    api_images = get_all_api_images()
    print(f"Found {len(api_images)} API images to evaluate")
    
    # Load existing results to avoid reprocessing
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
        
        # Skip if already processed
        if base_name in processed_ids:
            print(f"\nSkipping {i+1}/{len(api_images)}: {api_image} (already processed)")
            skipped += 1
            continue
            
        print(f"\nProcessing {i+1}/{len(api_images)}: {api_image}")
        print("🔄 Processing both OFFLINE and ONLINE settings...")
        
        start_time = time.time()
        evaluation = process_single_evaluation(api_image)
        end_time = time.time()
        processing_time = end_time - start_time
        
        evaluations.append(evaluation)
        
        if evaluation is not None:
            print(f"✓ Successfully evaluated {api_image} (took {processing_time:.1f}s)")
            print(f"📊 Results include both offline and online factor results")
            # Save immediately after each successful evaluation
            save_to_jsonl([evaluation], OUTPUT_JSONL_PATH)
            print(f"💾 Results saved to: {OUTPUT_JSONL_PATH}")
        else:
            # Check if it was a timeout (processing time close to 5 minutes)
            if processing_time >= 290:  # Close to 5 minutes (300 seconds)
                timeout_count += 1
                print(f"⏰ Timeout: {api_image} (took {processing_time:.1f}s)")
            else:
                print(f"✗ Failed to evaluate {api_image} (took {processing_time:.1f}s)")
        
        # Add delay between requests to avoid rate limiting
        if i < len(api_images) - 1:  # Don't wait after the last request
            print("Waiting 10 seconds before next request...")
            time.sleep(10)
    
    # Print summary
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
    print("📊 Each result includes both offline_factor_results and online_factor_results")

if __name__ == "__main__":
    main()

