# AuralEyes - README.md

## Overview

**AuralEyes** is an AI-powered application designed to assist visually impaired individuals by providing real-time insights about their surroundings. It leverages advanced machine learning models, natural language processing, and computer vision to enable users to understand scenes, detect objects and obstacles, and convert visual content into speech.

---

## Features

- **Scene Understanding**: Generates descriptive and detailed textual outputs to help users comprehend scenes.
- **Text-to-Speech**: Reads text detected in uploaded images aloud for better accessibility.
- **Object & Obstacle Detector**: Identifies objects and obstacles in images, enhancing user safety.
- **Personalized Assistance**: Tailors specific guidance based on uploaded images for daily tasks.

---

## <div align="center">AuralEyes Flow chart </div>
<div align="center"><img src ="https://github.com/user-attachments/assets/4b698fd3-2071-4f91-9c3b-b90fafc3abc3"></div>

## Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.7+
- Streamlit
- Google Generative AI (`google.generativeai`)
- LangChain
- EasyOCR
- Pillow
- Matplotlib
- gTTS
- IPython

### Setup

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install streamlit google-generativeai langchain easyocr pillow matplotlib gtts ipython
3. Set your Google Generative AI API key in the environment:
   ```bash
   genai.configure(api_key="YOUR_GOOGLE_API_KEY")


### Usage

1. Run the Streamlit application
   ```bash
   streamlit run app.py
2. Open the app in your browser (default: `http://localhost:8501`).
3. Upload an image to begin.

## How It Works
### Application Flow
1. **Upload an Image:**
-   Users upload an image via the file uploader.
-   The uploaded image is displayed for reference.

2.  **Select a Feature:**

-  **Scene Understanding:**
    -   Generates a detailed textual description of the uploaded image.
    -   Converts the description into speech using Text-to-Speech (TTS).
-  **Text-to-Speech:**
    -   Detects and reads text within the uploaded image using EasyOCR.
- **Object & Obstacle Detector:**
    -   Identifies objects and obstacles in the image for safe navigation.
    -   Provides a detailed textual output and audio feedback.
-  **Personalized Assistance:**
    -   Offers guidance for daily tasks based on image content.
    -   Converts insights into speech for ease of use.
3.  **Session Memory:**
-   Tracks completed tasks and displays them in the sidebar for reference.
### Key Components
- **Streamlit Interface:**

    -   User-friendly interface with an intuitive layout.
    -   Features a sidebar for session memory and feature selection.
- **Google Generative AI:**

    -   Utilized for generating context-specific descriptions of images.
-  **EasyOCR:**

    -   Extracts text from images to enhance accessibility.
-   **Text-to-Speech:**

    -   Converts textual insights into audio feedback.
-   **LangChain:**

    -   Handles structured prompt formatting for seamless AI interaction.

## Code Structure
### Import Libraries
The application integrates multiple libraries for various functionalities:

-   `google.generativeai:` For generating descriptions using Google's Generative AI.
-   `easyocr:` For text extraction from images.
-   `gTTS:` For converting text to speech.
-   `Streamlit:`For building an interactive web app.
### Key Functions
-   `update_memory(task, result)`: Updates session memory with completed tasks and results.

-   **Google Generative AI Runnable:** Custom `Runnable` class wraps Google's Generative AI for generating image-based insights.

-   **Feature Buttons:** Each feature button triggers a specific task (e.g., Scene Understanding, Text-to-Speech).

# <div align="center">AuralEyes UI</div>
<div align="center"><img src ="https://github.com/user-attachments/assets/6413050e-7087-4394-a3e8-28f9a20f01d9"></div>
<div align="center"><img src ="https://github.com/user-attachments/assets/e82796f7-5351-4008-9241-a085be26c791"></div>

## Future Enhancements
-   **Real-Time Object Detection:** Integrating live camera feed for dynamic obstacle detection.
-   **Multilingual Support:** Extending text-to-speech capabilities to multiple languages.
-   **Enhanced Accessibility:** Adding voice commands for a hands-free experience.



## License
Copyright (c) 2024 Ajala Oluwafemi daniel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
