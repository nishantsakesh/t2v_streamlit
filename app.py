# import streamlit as st
# import cv2
# import numpy as np
# import torch
# from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler, AutoPipelineForImage2Image
# from PIL import Image
# import os
# import urllib.request
# from segment_anything import SamPredictor, sam_model_registry
# from transformers import CLIPProcessor, CLIPModel
# import gc
# import logging
# import random
# import time
# import re
# import shutil

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # --- Utility Functions ---

# # Global placeholders for download progress, so they can be updated by reporthook
# # These will be initialized in the main app flow
# download_progress_bar = None
# download_status_text = None
# download_start_time = None

# def download_sam_checkpoint(checkpoint_path="sam_vit_h_4b8939.pth"):
#     global download_progress_bar, download_status_text, download_start_time

#     sam_checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

#     if not os.path.exists(checkpoint_path):
#         logging.info("Downloading SAM model checkpoint...")

#         # Initialize placeholders for progress bar and status text
#         download_status_text = st.empty()
#         download_progress_bar = st.progress(0)
        
#         download_status_text.info("Downloading SAM model... (This is a one-time download)")
#         download_start_time = time.time()

#         def reporthook(block_num, block_size, total_size):
#             """
#             Callback function for urllib.request.urlretrieve to report download progress.
#             """
#             global download_progress_bar, download_status_text, download_start_time

#             downloaded_bytes = block_num * block_size
            
#             if total_size > 0:
#                 percent = min(int(downloaded_bytes * 100 / total_size), 100)
#                 download_progress_bar.progress(percent)

#                 elapsed_time = time.time() - download_start_time
#                 if elapsed_time > 0:
#                     download_speed_bps = downloaded_bytes / elapsed_time # bytes per second
#                     remaining_bytes = total_size - downloaded_bytes
                    
#                     if download_speed_bps > 0:
#                         time_left_seconds = remaining_bytes / download_speed_bps
#                         minutes, seconds = divmod(int(time_left_seconds), 60)
#                         time_left_str = f"{minutes}m {seconds}s"
#                     else:
#                         time_left_str = "Calculating..." # Speed is 0 at very start
#                 else:
#                     time_left_str = "Calculating..." # Elapsed time is 0 at very start

#                 download_status_text.info(
#                     f"Downloading SAM model... {percent}% ({downloaded_bytes / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB) - Estimated time remaining: {time_left_str}"
#                 )
#             else:
#                 download_status_text.info(f"Downloading SAM model... {downloaded_bytes / (1024*1024):.2f} MB")

#         try:
#             urllib.request.urlretrieve(sam_checkpoint_url, checkpoint_path, reporthook=reporthook)
#             logging.info("SAM model download complete!")
#             download_status_text.empty() # Clear the message after download
#             download_progress_bar.empty() # Clear the progress bar
#             st.success("SAM model downloaded successfully!") # Success message
#         except Exception as e:
#             download_status_text.error(f"Error downloading SAM model: {e}")
#             logging.error(f"Error downloading SAM model: {e}")
#             download_progress_bar.empty() # Clear the progress bar
#     else:
#         logging.info("SAM model checkpoint already exists.")
#         st.info("SAM model already exists.") # Inform user if already exists

# @st.cache_resource # Cache the model to avoid reloading on every rerun
# def load_sam_model(model_type="vit_h", checkpoint="sam_vit_h_4b8939.pth"):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     sam = sam_model_registry[model_type](checkpoint=checkpoint)
#     sam.to(device=device)
#     predictor = SamPredictor(sam)
#     return predictor

# @st.cache_resource # Cache the CLIP model
# def load_clip_model():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
#     clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     return clip_model, clip_processor

# @st.cache_resource # Cache the Stable Diffusion pipeline
# def load_img2img_pipeline(model_choice="Standard Stable Diffusion"):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     if model_choice == "Standard Stable Diffusion":
#         st.info("Loading Standard Stable Diffusion v1.5 pipeline...")
#         pipe = AutoPipelineForImage2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 if device == "cuda" else torch.float32)
#         pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
#         pipe.to(device)
#         return pipe, 50 # Default inference steps for standard SD
    
#     elif model_choice == "Fast (LCM-LoRA)":
#         st.info("Loading Stable Diffusion with LCM-LoRA for faster generation...")
#         pipe = AutoPipelineForImage2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 if device == "cuda" else torch.float32)
#         pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
#         pipe.fuse_lora()
#         pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
#         pipe.to(device)
#         return pipe, 4 # LCM typically needs very few steps (e.g., 4-8)

# def extract_foreground_with_sam(image, predictor, point_coords=None):
#     image_np = np.array(image)
#     predictor.set_image(image_np)
#     if point_coords is None:
#         point_coords = np.array([[image.width // 2, image.height // 2]])
#     masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=np.array([1]))
#     mask = masks[0]
#     return mask

# def blend_foreground_background(foreground, background, mask):
#     foreground = foreground.convert("RGBA")
#     background = background.convert("RGBA")
#     mask = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
#     blended = Image.composite(foreground, background, mask)
#     return blended.convert("RGB")

# def transform_image(image, prompt, img2img_pipe, strength=0.45, num_inference_steps=50):
#     with torch.inference_mode():
#         transformed_image = img2img_pipe(prompt=prompt, image=image, strength=strength, num_inference_steps=num_inference_steps).images[0]
#     return transformed_image

# def create_video(image_folder, output_video, fps=5):
#     images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
#     images.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
#     if not images:
#         logging.error(f"No images found in {image_folder}")
#         return
#     frame = cv2.imread(os.path.join(image_folder, images[0]))
#     height, width, layers = frame.shape
#     video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
#     for image in images:
#         video.write(cv2.imread(os.path.join(image_folder, image)))
#     cv2.destroyAllWindows()
#     video.release()

# def validate_generated_frame(image, prompt, clip_model, clip_processor):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
#     with torch.no_grad():
#         outputs = clip_model(**inputs).logits_per_image.item()
#     return outputs

# def generate_story_prompts(initial_prompt, num_frames):
#     story_prompts = [initial_prompt]
#     keywords = re.findall(r'\b\w+\b', initial_prompt.lower())

#     if not keywords:
#         keywords = ["scene", "image", "visuals", "story", "elements"]

#     camera_moves = ["a close-up shot", "a wide shot", "a panning shot", "a tracking shot", "a dolly shot", "a bird's-eye view", "a low-angle shot"]
#     motion_verbs = ["moves", "shifts", "glides", "jumps", "flies", "rotates", "grows", "shrinks", "appears", "disappears",
#                      "accelerates", "decelerates", "tilts", "veers", "soars", "plunges", "spirals", "surges"]
#     motion_modifiers = ["slightly", "rapidly", "smoothly", "abruptly", "gradually", "in a circular motion",
#                          "with increasing speed", "with a sudden jolt", "in a flowing manner", "in a jerky motion"]
#     direction_modifiers = ["upwards", "downwards", "leftwards", "rightwards", "forwards", "backwards",
#                              "diagonally", "in a straight line", "in a curved path"]
#     action_events = ["undergoes a dramatic change", "interacts with another object", "transforms into something else", "reveals a hidden detail", "initiates a sequence of actions", "reaches a climax", "displays a captivating visual effect"]
#     scene_descriptions = ["a vibrant and dynamic scene", "a serene and tranquil setting", "a mysterious and enigmatic atmosphere", "a chaotic and intense environment", "a surreal and dreamlike sequence", "a realistic and detailed depiction"]
#     narrative_transitions = ["then", "subsequently", "the focus shifts to", "meanwhile", "a new element emerges", "the scene evolves", "a dramatic turn occurs"]

#     used_combinations = set()
#     variation_weights = [0.35, 0.3, 0.2, 0.15]

#     for i in range(num_frames - 1):
#         variation_type = random.choices(["motion", "action", "scene", "narrative"], weights=variation_weights)[0]

#         if variation_type == "motion":
#             camera_move = random.choice(camera_moves)
#             motion_verb = random.choice(motion_verbs)
#             motion_modifier = random.choice(motion_modifiers)
#             direction_modifier = random.choice(direction_modifiers)
#             combination = (camera_move, motion_verb, motion_modifier, direction_modifier)

#             if combination in used_combinations:
#                 variation_type = random.choice(["action", "scene", "narrative"])
#             else:
#                 used_combinations.add(combination)
#                 new_prompt = f"{camera_move}, {initial_prompt}, it {motion_verb} {motion_modifier} {direction_modifier}."

#         if variation_type == "action":
#             action_event = random.choice(action_events)
#             new_prompt = f"{initial_prompt}, it {action_event}."

#         elif variation_type == "scene":
#             scene_description = random.choice(scene_descriptions)
#             new_prompt = f"{scene_description}, {initial_prompt}."

#         elif variation_type == "narrative":
#             narrative_transition = random.choice(narrative_transitions)
#             new_prompt = f"{narrative_transition}, {initial_prompt}."

#         if len(new_prompt.split()) > 20:
#             parts = new_prompt.split(",")
#             if len(parts) > 1:
#                 new_prompt = ", ".join(parts[:2]) + ". " + ", ".join(parts[2:])

#         story_prompts.append(new_prompt)
#         logging.info(f"Generated Prompt {i+2}: {new_prompt}")

#     return story_prompts


# def generate_story_video(image_file, initial_prompt, target_resolution, sam_predictor, clip_model, clip_processor, fps, duration_secs, model_choice, img2img_pipe_tuple, strength=0.45):
#     logging.info("Starting generate_story_video function...")
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     output_video_path = "./story_video.mp4"

#     try:
#         original_image = Image.open(image_file).convert("RGB")
#         resized_image = original_image.resize(target_resolution, Image.Resampling.LANCZOS)

#         similarity = validate_generated_frame(resized_image, initial_prompt, clip_model, clip_processor)
#         if similarity < 0.2:
#             st.warning(f"Image and text prompt are not very similar, similarity: {similarity:.2f}. Please try a different image or prompt.")
#             return None

#         num_frames = int(fps * duration_secs)
#         story_prompts = generate_story_prompts(initial_prompt, num_frames)

#         # Unpack the pipeline and inference steps
#         img2img_pipe, num_inference_steps = img2img_pipe_tuple
        
#         output_folder = "./story_frames"
#         os.makedirs(output_folder, exist_ok=True)

#         previous_frame = resized_image
#         object_mask = extract_foreground_with_sam(previous_frame, sam_predictor)
#         mask_shift_x = 0
#         mask_shift_y = 0
#         mask_increment = 5

#         progress_text = st.empty()
#         progress_bar = st.progress(0)

#         for i, prompt in enumerate(story_prompts):
#             progress_text.text(f"Generating frame {i + 1}/{num_frames} with prompt: {prompt}")
#             progress_bar.progress((i + 1) / num_frames)

#             start_time = time.time()
#             transformed_frame = transform_image(previous_frame, prompt, img2img_pipe, strength, num_inference_steps)
#             end_time = time.time()
#             generation_time = end_time - start_time

#             similarity = validate_generated_frame(transformed_frame, prompt, clip_model, clip_processor)
#             if similarity < 0.2:
#                 logging.warning(f"Frame {i + 1} rejected due to low similarity ({similarity:.2f}). Retrying...")
#                 continue

#             mask_shift_x += random.randint(-mask_increment, mask_increment)
#             mask_shift_y += random.randint(-mask_increment, mask_increment)
#             shifted_mask = np.roll(object_mask, mask_shift_x, axis=1)
#             shifted_mask = np.roll(shifted_mask, mask_shift_y, axis=0)

#             masked_frame = blend_foreground_background(transformed_frame, previous_frame, shifted_mask)
#             masked_frame.save(os.path.join(output_folder, f"frame_{i:04d}.png"))
#             logging.info(f"Saved frame {i + 1}/{len(story_prompts)} with similarity {similarity:.2f} and generation time: {generation_time:.2f} seconds")

#             previous_frame = transformed_frame

#             del transformed_frame
#             gc.collect()
#             torch.cuda.empty_cache()
#             if i >= len(story_prompts) - 1:
#                 break
        
#         progress_text.text("Creating video...")
#         create_video(output_folder, output_video_path, fps)
#         logging.info("Story video generation complete!")
#         return output_video_path

#     except FileNotFoundError:
#         st.error(f"Image file not found.")
#         return None
#     except Exception as e:
#         st.error(f"Error generating video: {e}")
#         logging.error(f"Error generating video: {e}")
#         return None
#     finally:
#         if os.path.exists("./story_frames"):
#             shutil.rmtree("./story_frames")
#             logging.info("story_frames folder cleaned up.")


# # --- Streamlit App Layout ---

# st.set_page_config(page_title="Image to Video Generator", layout="centered")

# st.title("🎬 Image to Video Generator (Optimized)")
# st.markdown("""
# Upload an image, provide a text prompt, and specify the desired frames per second (FPS) and duration to generate a video!
# You can now choose between a standard model and a faster, optimized model (LCM-LoRA).
# """)

# # --- Model Pre-loading Section ---
# # These functions will run as soon as the page loads and cache the models.
# # The st.spinner/st.info messages here will show during the initial load.

# with st.spinner("Initializing application..."):
#     download_sam_checkpoint()
#     sam_predictor = load_sam_model()
#     clip_model, clip_processor = load_clip_model()

# # --- End Model Pre-loading Section ---

# # Input fields
# uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "png", "jpeg"])
# prompt = st.text_input("Enter the initial text prompt:", "A futuristic city with flying cars")
# fps = st.slider("Frames per second (FPS):", min_value=1, max_value=30, value=10)
# duration_secs = st.slider("Video Duration (seconds):", min_value=1, max_value=10, value=3)

# # New: Model selection dropdown
# model_choice = st.selectbox(
#     "Choose Model Optimization:",
#     ("Standard Stable Diffusion", "Fast (LCM-LoRA)"),
#     help="Standard Stable Diffusion provides good quality. Fast (LCM-LoRA) generates videos much quicker with slightly less detail."
# )

# # Load the img2img pipeline based on the selected model choice
# # This will also be cached, so it only reloads if model_choice changes
# img2img_pipe_tuple = load_img2img_pipeline(model_choice)


# # Resolution setting (fixed for now as per notebook)
# target_resolution = (512, 512)

# if uploaded_file is not None:
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

#     if st.button("Generate Video"):
#         if uploaded_file is not None and prompt:
#             with st.spinner("Generating video frames..."):
#                 video_path = generate_story_video(
#                     uploaded_file,
#                     prompt,
#                     target_resolution,
#                     sam_predictor,
#                     clip_model,
#                     clip_processor,
#                     fps,
#                     duration_secs,
#                     model_choice,
#                     img2img_pipe_tuple # Pass the loaded pipeline tuple
#                 )

#             if video_path and os.path.exists(video_path):
#                 st.success("Video generated successfully!")
#                 st.video(video_path)
#                 with open(video_path, "rb") as file:
#                     btn = st.download_button(
#                         label="Download Video",
#                         data=file,
#                         file_name="story_video.mp4",
#                         mime="video/mp4"
#                     )
#                 try:
#                     os.remove(video_path)
#                     logging.info(f"Video file cleaned up: {video_path}")
#                 except OSError as e:
#                     logging.error(f"Error removing video file {video_path}: {e}")
#             else:
#                 st.error("Failed to generate video. Please check the logs for more details.")
#         else:
#             st.warning("Please upload an image and enter a prompt to generate the video.")

# st.markdown("---")
# st.markdown("Developed as a final year major project.")




import streamlit as st
import cv2
import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler, AutoPipelineForImage2Image
from PIL import Image
import os
import urllib.request
from segment_anything import SamPredictor, sam_model_registry
from transformers import CLIPProcessor, CLIPModel
import gc
import logging
import random
import time
import re
import shutil # For cleaning up temporary files

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Utility Functions ---

# Global placeholders for download progress, so they can be updated by reporthook
# These will be initialized in the main app flow
download_progress_bar = None
download_status_text = None
download_start_time = None

def download_sam_checkpoint(checkpoint_path="sam_vit_h_4b8939.pth"):
    global download_progress_bar, download_status_text, download_start_time

    sam_checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

    if not os.path.exists(checkpoint_path):
        logging.info("Downloading SAM model checkpoint...")

        # Initialize placeholders for progress bar and status text
        download_status_text = st.empty()
        download_progress_bar = st.progress(0)
        
        download_status_text.info("Downloading SAM model... (This is a one-time download)")
        download_start_time = time.time()

        def reporthook(block_num, block_size, total_size):
            """
            Callback function for urllib.request.urlretrieve to report download progress.
            """
            global download_progress_bar, download_status_text, download_start_time

            downloaded_bytes = block_num * block_size
            
            if total_size > 0:
                percent = min(int(downloaded_bytes * 100 / total_size), 100)
                download_progress_bar.progress(percent)

                elapsed_time = time.time() - download_start_time
                if elapsed_time > 0:
                    download_speed_bps = downloaded_bytes / elapsed_time # bytes per second
                    remaining_bytes = total_size - downloaded_bytes
                    
                    if download_speed_bps > 0:
                        time_left_seconds = remaining_bytes / download_speed_bps
                        minutes, seconds = divmod(int(time_left_seconds), 60)
                        time_left_str = f"{minutes}m {seconds}s"
                    else:
                        time_left_str = "Calculating..." # Speed is 0 at very start
                else:
                    time_left_str = "Calculating..." # Elapsed time is 0 at very start

                download_status_text.info(
                    f"Downloading SAM model... {percent}% ({downloaded_bytes / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB) - Estimated time remaining: {time_left_str}"
                )
            else:
                download_status_text.info(f"Downloading SAM model... {downloaded_bytes / (1024*1024):.2f} MB")

        try:
            urllib.request.urlretrieve(sam_checkpoint_url, checkpoint_path, reporthook=reporthook)
            logging.info("SAM model download complete!")
            download_status_text.empty() # Clear the message after download
            download_progress_bar.empty() # Clear the progress bar
            st.success("SAM model downloaded successfully!") # Success message
        except Exception as e:
            download_status_text.error(f"Error downloading SAM model: {e}")
            logging.error(f"Error downloading SAM model: {e}")
            download_progress_bar.empty() # Clear the progress bar
    else:
        logging.info("SAM model checkpoint already exists.")
        st.info("SAM model already exists.") # Inform user if already exists

@st.cache_resource # Cache the model to avoid reloading on every rerun
def load_sam_model(model_type="vit_h", checkpoint="sam_vit_h_4b8939.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

@st.cache_resource # Cache the CLIP model
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor

@st.cache_resource # Cache the Stable Diffusion pipeline
def load_img2img_pipeline(model_choice="Standard Stable Diffusion"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model_choice == "Standard Stable Diffusion":
        st.info("Loading Standard Stable Diffusion v1.5 pipeline...")
        pipe = AutoPipelineForImage2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(device)
        return pipe, 50 # Default inference steps for standard SD
    
    elif model_choice == "Fast (LCM-LoRA)":
        st.info("Loading Stable Diffusion with LCM-LoRA for faster generation...")
        pipe = AutoPipelineForImage2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        pipe.fuse_lora()
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(device)
        return pipe, 4 # LCM typically needs very few steps (e.g., 4-8)

def extract_foreground_with_sam(image, predictor, point_coords=None):
    image_np = np.array(image)
    predictor.set_image(image_np)
    if point_coords is None:
        point_coords = np.array([[image.width // 2, image.height // 2]])
    masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=np.array([1]))
    mask = masks[0]
    return mask

def blend_foreground_background(foreground, background, mask):
    foreground = foreground.convert("RGBA")
    background = background.convert("RGBA")
    mask = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
    blended = Image.composite(foreground, background, mask)
    return blended.convert("RGB")

def transform_image(image, prompt, img2img_pipe, strength=0.45, num_inference_steps=50):
    with torch.inference_mode():
        transformed_image = img2img_pipe(prompt=prompt, image=image, strength=strength, num_inference_steps=num_inference_steps).images[0]
    return transformed_image

def create_video(image_folder, output_video, fps=5):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    if not images:
        logging.error(f"No images found in {image_folder}")
        return
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()

def validate_generated_frame(image, prompt, clip_model, clip_processor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs).logits_per_image.item()
    return outputs

def generate_story_prompts(initial_prompt, num_frames):
    story_prompts = [initial_prompt]
    keywords = re.findall(r'\b\w+\b', initial_prompt.lower())

    if not keywords:
        keywords = ["scene", "image", "visuals", "story", "elements"]

    camera_moves = ["a close-up shot", "a wide shot", "a panning shot", "a tracking shot", "a dolly shot", "a bird's-eye view", "a low-angle shot"]
    motion_verbs = ["moves", "shifts", "glides", "jumps", "flies", "rotates", "grows", "shrinks", "appears", "disappears",
                     "accelerates", "decelerates", "tilts", "veers", "soars", "plunges", "spirals", "surges"]
    motion_modifiers = ["slightly", "rapidly", "smoothly", "abruptly", "gradually", "in a circular motion",
                         "with increasing speed", "with a sudden jolt", "in a flowing manner", "in a jerky motion"]
    direction_modifiers = ["upwards", "downwards", "leftwards", "rightwards", "forwards", "backwards",
                             "diagonally", "in a straight line", "in a curved path"]
    action_events = ["undergoes a dramatic change", "interacts with another object", "transforms into something else", "reveals a hidden detail", "initiates a sequence of actions", "reaches a climax", "displays a captivating visual effect"]
    scene_descriptions = ["a vibrant and dynamic scene", "a serene and tranquil setting", "a mysterious and enigmatic atmosphere", "a chaotic and intense environment", "a surreal and dreamlike sequence", "a realistic and detailed depiction"]
    narrative_transitions = ["then", "subsequently", "the focus shifts to", "meanwhile", "a new element emerges", "the scene evolves", "a dramatic turn occurs"]

    used_combinations = set()
    variation_weights = [0.35, 0.3, 0.2, 0.15]

    for i in range(num_frames - 1):
        variation_type = random.choices(["motion", "action", "scene", "narrative"], weights=variation_weights)[0]

        if variation_type == "motion":
            camera_move = random.choice(camera_moves)
            motion_verb = random.choice(motion_verbs)
            motion_modifier = random.choice(motion_modifiers)
            direction_modifier = random.choice(direction_modifiers)
            combination = (camera_move, motion_verb, motion_modifier, direction_modifier)

            if combination in used_combinations:
                variation_type = random.choice(["action", "scene", "narrative"])
            else:
                used_combinations.add(combination)
                new_prompt = f"{camera_move}, {initial_prompt}, it {motion_verb} {motion_modifier} {direction_modifier}."

        if variation_type == "action":
            action_event = random.choice(action_events)
            new_prompt = f"{initial_prompt}, it {action_event}."

        elif variation_type == "scene":
            scene_description = random.choice(scene_descriptions)
            new_prompt = f"{scene_description}, {initial_prompt}."

        elif variation_type == "narrative":
            narrative_transition = random.choice(narrative_transitions)
            new_prompt = f"{narrative_transition}, {initial_prompt}."

        if len(new_prompt.split()) > 20:
            parts = new_prompt.split(",")
            if len(parts) > 1:
                new_prompt = ", ".join(parts[:2]) + ". " + ", ".join(parts[2:])

        story_prompts.append(new_prompt)
        logging.info(f"Generated Prompt {i+2}: {new_prompt}")

    return story_prompts


def generate_story_video(image_file, initial_prompt, target_resolution, sam_predictor, clip_model, clip_processor, fps, duration_secs, model_choice, img2img_pipe_tuple, strength=0.45):
    logging.info("Starting generate_story_video function...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_video_path = "./story_video.mp4"

    try:
        original_image = Image.open(image_file).convert("RGB")
        resized_image = original_image.resize(target_resolution, Image.Resampling.LANCZOS)

        similarity = validate_generated_frame(resized_image, initial_prompt, clip_model, clip_processor)
        if similarity < 0.2:
            st.warning(f"Image and text prompt are not very similar, similarity: {similarity:.2f}. Please try a different image or prompt.")
            return None

        num_frames = int(fps * duration_secs)
        story_prompts = generate_story_prompts(initial_prompt, num_frames)

        # Unpack the pipeline and inference steps
        img2img_pipe, num_inference_steps = img2img_pipe_tuple
        
        output_folder = "./story_frames"
        os.makedirs(output_folder, exist_ok=True)

        previous_frame = resized_image
        object_mask = extract_foreground_with_sam(previous_frame, sam_predictor)
        mask_shift_x = 0
        mask_shift_y = 0
        mask_increment = 5

        progress_text = st.empty()
        progress_bar = st.progress(0)

        for i, prompt in enumerate(story_prompts):
            progress_text.text(f"Generating frame {i + 1}/{num_frames} with prompt: {prompt}")
            progress_bar.progress((i + 1) / num_frames)

            start_time = time.time()
            transformed_frame = transform_image(previous_frame, prompt, img2img_pipe, strength, num_inference_steps)
            end_time = time.time()
            generation_time = end_time - start_time

            similarity = validate_generated_frame(transformed_frame, prompt, clip_model, clip_processor)
            if similarity < 0.2:
                logging.warning(f"Frame {i + 1} rejected due to low similarity ({similarity:.2f}). Retrying...")
                continue

            mask_shift_x += random.randint(-mask_increment, mask_increment)
            mask_shift_y += random.randint(-mask_increment, mask_increment)
            shifted_mask = np.roll(object_mask, mask_shift_x, axis=1)
            shifted_mask = np.roll(shifted_mask, mask_shift_y, axis=0)

            masked_frame = blend_foreground_background(transformed_frame, previous_frame, shifted_mask)
            masked_frame.save(os.path.join(output_folder, f"frame_{i:04d}.png"))
            logging.info(f"Saved frame {i + 1}/{len(story_prompts)} with similarity {similarity:.2f} and generation time: {generation_time:.2f} seconds")

            previous_frame = transformed_frame

            del transformed_frame
            gc.collect()
            torch.cuda.empty_cache()
            if i >= len(story_prompts) - 1:
                break
        
        progress_text.text("Creating video...")
        create_video(output_folder, output_video_path, fps)
        logging.info("Story video generation complete!")
        return output_video_path

    except FileNotFoundError:
        st.error(f"Image file not found.")
        return None
    except Exception as e:
        st.error(f"Error generating video: {e}")
        logging.error(f"Error generating video: {e}")
        return None
    finally:
        if os.path.exists("./story_frames"):
            shutil.rmtree("./story_frames")
            logging.info("story_frames folder cleaned up.")


# --- Streamlit App Layout ---

st.set_page_config(page_title="Image to Video Generator", layout="centered")

st.title("🎬 Image to Video Generator (Optimized)")
st.markdown("""
Upload an image, provide a text prompt, and specify the desired frames per second (FPS) and duration to generate a video!
You can now choose between a standard model and a faster, optimized model (LCM-LoRA).
""")

# --- Model Pre-loading Section ---
# These functions will run as soon as the page loads and cache the models.
# The st.spinner/st.info messages here will show during the initial load.

with st.spinner("Initializing application..."):
    download_sam_checkpoint()
    sam_predictor = load_sam_model()
    clip_model, clip_processor = load_clip_model()

# --- End Model Pre-loading Section ---

# Input fields
uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "png", "jpeg"])
prompt = st.text_input("Enter the initial text prompt:", "A futuristic city with flying cars")
fps = st.slider("Frames per second (FPS):", min_value=1, max_value=30, value=10)
duration_secs = st.slider("Video Duration (seconds):", min_value=1, max_value=10, value=3)

# New: Model selection dropdown
model_choice = st.selectbox(
    "Choose Model Optimization:",
    ("Standard Stable Diffusion", "Fast (LCM-LoRA)"),
    help="Standard Stable Diffusion provides good quality. Fast (LCM-LoRA) generates videos much quicker with slightly less detail."
)

# Load the img2img pipeline based on the selected model choice
# This will also be cached, so it only reloads if model_choice changes
img2img_pipe_tuple = load_img2img_pipeline(model_choice)


# Resolution setting (fixed for now as per notebook)
target_resolution = (512, 512)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Video"):
        if uploaded_file is not None and prompt:
            with st.spinner("Generating video frames..."):
                video_path = generate_story_video(
                    uploaded_file,
                    prompt,
                    target_resolution,
                    sam_predictor,
                    clip_model,
                    clip_processor,
                    fps,
                    duration_secs,
                    model_choice,
                    img2img_pipe_tuple # Pass the loaded pipeline tuple
                )

            if video_path and os.path.exists(video_path):
                st.success("Video generated successfully!")
                st.video(video_path)
                with open(video_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Video",
                        data=file,
                        file_name="story_video.mp4",
                        mime="video/mp4"
                    )
                try:
                    os.remove(video_path)
                    logging.info(f"Video file cleaned up: {video_path}")
                except OSError as e:
                    logging.error(f"Error removing video file {video_path}: {e}")
            else:
                st.error("Failed to generate video. Please check the logs for more details.")
        else:
            st.warning("Please upload an image and enter a prompt to generate the video.")

st.markdown("---")
st.markdown("Developed as a final year major project.")