import base64
import shutil
import streamlit as st
from openai import OpenAI
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.tasks.task_literals import InputType, OutputType
from lyzr_automata.pipelines.linear_sync_pipeline  import  LinearSyncPipeline
from dotenv import load_dotenv,find_dotenv
from PIL import Image
import os

load_dotenv(find_dotenv())
api = os.getenv("OPENAI_API_KEY")

client = OpenAI()

st.set_page_config(
    page_title="Lyzr Art Analyzer",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto",
    page_icon="lyzr-logo-cut.png",
)

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Lyzr Art Analyzer")
st.markdown("### Welcome to the Lyzr Art Analyzer!")

def remove_existing_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)

data_directory = "data"
os.makedirs(data_directory, exist_ok=True)
remove_existing_files(data_directory)


uploaded_file = st.sidebar.file_uploader("Choose PDF file", type=["jpg",'png','webp'])

if uploaded_file is not None:
    # Save the uploaded PDF file to the data directory
    file_path = os.path.join(data_directory, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getvalue())

    # Display the path of the stored file
    st.sidebar.success(f"File successfully saved")

def get_all_files(data_directory):
    # List to store all file paths
    file_paths = []

    # Walk through the directory tree
    for root, dirs, files in os.walk(data_directory):
        for file in files:
            # Join the root path with the file name to get the absolute path
            file_path = os.path.join(root, file)
            # Append the file path to the list
            file_paths.append(file_path)
            break

    return file_paths


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        base64_encoded = base64.b64encode(image_bytes)
        return base64_encoded.decode('utf-8')

def generate_img_insights(image):
    response = client.chat.completions.create(
      model="gpt-4-vision-preview",
      messages=[
        {
          "role": "user",
          "content": [
            {"type": "text", "text": """Describe image style and give insights of an image.give in below format:
            Art Style: 
            Insights:
            """},
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
              },
            },
          ],
        }
      ],
      max_tokens=300,
    )

    return response.choices[0].message.content

def generate_image(insights):
    artist_agent = Agent(
            prompt_persona="You are an Artist and you generate image based on art style and insights",
            role="Artist",
        )

    open_ai_model_image = OpenAIModel(
        api_key=api,
        parameters={
            "n": 1,
            "model": "dall-e-3",
        },
    )

    art_generation_task = Task(
            name="Art Image Creation",
            output_type=OutputType.IMAGE,
            input_type=InputType.TEXT,
            model=open_ai_model_image,
            log_output=True,
            instructions=f"""Generate an image with below instructions:
            {insights}
            """,
        )

    output = LinearSyncPipeline(
            name="Generate Art",
            completion_message="Art Generated!",
            tasks=[
                art_generation_task
            ],
        ).run()
    return output[0]['task_output'].url

path = get_all_files(data_directory)
if path is not None:
    for i in path:
        st.sidebar.image(i)
        base64_image = image_to_base64(i)
        insights = generate_img_insights(base64_image)
        images = generate_image(insights)
        if images:
            st.markdown(insights)
            st.image(images)





