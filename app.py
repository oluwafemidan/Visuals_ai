import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import easyocr
from gtts import gTTS
from IPython.display import Audio


# Set up Streamlit page
st.set_page_config(page_title="AuralEyes")
st.title("👁️ AuralEyes - AI Assistant for Visually Impaired")
st.sidebar.title("⚙️ Features")
st.sidebar.markdown("""
- Scene Understanding
- Text-to-Speech
- Object & Obstacle Detector
- Personalized Assistance             
""")

# Initialize memory storage
if "memory" not in st.session_state:
    st.session_state.memory = []

# Function to update memory
def update_memory(task, result):
    st.session_state.memory.append({"task": task, "result": result})

# Display memory in the sidebar
st.sidebar.title("🧠 Session Memory")
if st.session_state.memory:
    for i, entry in enumerate(st.session_state.memory):
        st.sidebar.write(f"**Task {i + 1}: {entry['task']}**")
        st.sidebar.write(entry['result'])
else:
    st.sidebar.write("No tasks completed yet.")


api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)


# define LangChain Prompt Template
langchain_prompt = PromptTemplate(
    input_variables=["image_description"],
    template="""
    {image_description}
    """
)

# Creating of a custom Runnable to wrap the Google Generative AI model
class GoogleGenerativeModelRunnable(Runnable):
    def __init__(self, model):
        self.model = model

    def invoke(self, input):
        response = self.model.generate_content(
            input,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=8192, temperature=1, top_p=0.95
            ),
        )
        return response.text

# Initialize the Generative Model
model = genai.GenerativeModel(
    "gemini-1.5-flash-002",
    system_instruction="""
    You are an assistant designed to support visually impaired individuals in perceiving and interacting with their surroundings.
    Your role involves providing clear and actionable insights based on uploaded images to assist users in various tasks. Your responsibilities include:

    Real-Time Scene Understanding:
    Generate descriptive and detailed textual outputs that clearly interpret the content of the uploaded image, helping users comprehend the scene effectively.

    Object and Obstacle Detection for Safe Navigation:
    Identify and highlight objects or obstacles in the image, providing essential information to enhance user safety and situational awareness.

    Personalized Assistance for Daily Tasks:
    Offer specific guidance tailored to the user's needs based on the image. This includes recognizing items, reading labels,
    or delivering context-specific information to facilitate daily activities.
    """
)

# then wrap the model in the custom Runnable
runnable_model = GoogleGenerativeModelRunnable(model)



# Generate a description of the image
# Main app functionality
uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")

    task_1, task_2, task_3, task_4 = st.columns(4)
    if task_1.button("Scene Understanding", icon="🎬", use_container_width=True):
        prompt = """ Real-Time Scene Understanding:
        Generate descriptive and detailed textual outputs that clearly interpret the content of the uploaded image, helping users comprehend the scene effectively.
        """

        # Run the pipeline using LangChain's Runnable framework
        image_description = runnable_model.invoke([img, prompt])

        # Generate structured output using the prompt and LangChain
        response = langchain_prompt.format(image_description=image_description)

        st.write(response)
        update_memory("Scene Understanding", response)

        reader = easyocr.Reader(['en'])

        tts = gTTS(text=response, lang='en', slow=False)
        tts.save("output.mp3")

        st.audio("output.mp3", format="audio/mpeg", loop=True)



    if task_2.button("Text-to-Speech", icon="📢", use_container_width=True):

        try:
            reader = easyocr.Reader(['en'])
            
            # Reading text from the image
            text_list = reader.readtext(img, add_margin=0.55, width_ths=0.7, link_threshold=0.8, decoder='beamsearch', blocklist='=-', detail=0)
            
            if text_list:  # Check if text was detected
                text_comb = ' '.join(text_list)
                tts = gTTS(text=text_comb, lang='en', slow=False)
                tts.save("output.mp3")
                update_memory("Text To Speech", text_comb)
                
                st.audio("output.mp3", format="audio/mpeg", loop=True)
            else:
                raise ValueError("No text detected")
                
        except Exception as e:
            st.write("Unfortunately, no text was detected in the image.")
            tts = gTTS(text="Unfortunately, no text was detected in the image.", lang='en', slow=False)
            tts.save("output.mp3")
            
            st.audio("output.mp3", format="audio/mpeg", loop=True)
            update_memory("Text To Speech", "No text detected")

    

    if task_3.button("Object & Obstacle Detector", icon="🔍",use_container_width=True):
        prompt =  """Object and Obstacle Detection for Safe Navigation:
                Identify and highlight objects or obstacles in the image, providing essential information to enhance user safety and situational awareness.
                """

        # Run the pipeline using LangChain's Runnable framework
        image_description = runnable_model.invoke([img, prompt])

        # Generate structured output using the prompt and LangChain
        response = langchain_prompt.format(image_description=image_description)

        st.write(response)
        update_memory("Object and Obstacle Detection", response)

        reader = easyocr.Reader(['en'])

        tts = gTTS(text=response, lang='en', slow=False)
        tts.save("output.mp3")

        st.audio("output.mp3", format="audio/mpeg", loop=True)

    if task_4.button("Personalized Assistance", icon="🙋‍♀️", use_container_width=True):
        prompt = """ Personalized Assistance for Daily Tasks:
                Offer specific guidance tailored to the user's needs based on the image. This includes recognizing items, reading labels,
                or delivering context-specific information to facilitate daily activities. 
                """
        # Run the pipeline using LangChain's Runnable framework
        image_description = runnable_model.invoke([img, prompt])

        # Generate structured output using the prompt and LangChain
        response = langchain_prompt.format(image_description=image_description)

        st.write(response)
        update_memory("Personalized Assistance", response)

        reader = easyocr.Reader(['en'])

        tts = gTTS(text=response, lang='en', slow=False)
        tts.save("output.mp3")

        st.audio("output.mp3", format="audio/mpeg", loop=True)
    


