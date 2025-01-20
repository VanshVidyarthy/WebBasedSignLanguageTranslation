import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import os
import sys
import speech_recognition as sr
import subprocess

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

DEMO_VIDEO = 'demo.mp4'
DEMO_IMAGE = 'demo.jpg'
my_list=[]
# Sidebar styling
st.markdown(
    """
    <style>
    /* Adjust sidebar width */
    [data-testid="stSidebar"] > div:first-child {
        width: 400px;
        background-color: #f0f0f5; /* Light greyish background for the sidebar */
    }

    /* Adjust padding for the main content */
    .main {
        padding: 2rem;
        background-color: #ffffff; /* White background for the main content */
    }

    /* Customize the output text */
    .output-text {
        font-size: 1.2rem;
        color: #333;
    }

    /* Add a background color to the overall page */
    body {
        background-color: #e6e6fa; /* Light lavender background */
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.sidebar.title('Sign Language Detection')
st.sidebar.subheader('Parameters')

@st.cache_data()  # Use st.cache_data() instead of st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    if width is None and height is None:
        return image

    (h, w) = image.shape[:2]
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

app_mode = st.sidebar.selectbox('Choose the App Mode', ['About App', 'Sign Language to Text', 'Text to Sign Language'])

if app_mode == 'About App':
    st.markdown(
        """
        <main>
            <svg class="pl1" viewBox="0 0 128 128" width="160px" height="128px">
                <defs>
                    <linearGradient id="pl-grad" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0%" stop-color="#000" />
                        <stop offset="100%" stop-color="#fff" />
                    </linearGradient>
                    <mask id="pl-mask">
                        <rect x="0" y="0" width="128" height="128" fill="url(#pl-grad)" />
                    </mask>
                </defs>
                <g fill="var(--primary)">
                    <g class="pl1__g">
                        <g transform="translate(20,20) rotate(0,44,44)">
                            <g class="pl1__rect-g">
                                <rect class="pl1__rect" rx="8" ry="8" width="40" height="40" />
                                <rect class="pl1__rect" rx="8" ry="8" width="40" height="40" transform="translate(0,48)" />
                            </g>
                            <g class="pl1__rect-g" transform="rotate(180,44,44)">
                                <rect class="pl1__rect" rx="8" ry="8" width="40" height="40" />
                                <rect class="pl1__rect" rx="8" ry="8" width="40" height="40" transform="translate(0,48)" />
                            </g>
                        </g>
                    </g>
                </g>
                <g fill="hsl(343,90%,50%)" mask="url(#pl-mask)">
                    <g class="pl1__g">
                        <g transform="translate(20,20) rotate(0,44,44)">
                            <g class="pl1__rect-g">
                                <rect class="pl1__rect" rx="8" ry="8" width="40" height="40" />
                                <rect class="pl1__rect" rx="8" ry="8" width="40" height="40" transform="translate(0,48)" />
                            </g>
                            <g class="pl1__rect-g" transform="rotate(180,44,44)">
                                <rect class="pl1__rect" rx="8" ry="8" width="40" height="40" />
                                <rect class="pl1__rect" rx="8" ry="8" width="40" height="40" transform="translate(0,48)" />
                            </g>
                        </g>
                    </g>
                </g>
            </svg>
            <svg class="pl2" viewBox="0 0 128 128" width="128px" height="128px">
                <g fill="var(--primary)">
                    <g class="pl2__rect-g">
                        <rect class="pl2__rect" rx="8" ry="8" x="0" y="128" width="40" height="24" transform="rotate(180)" />
                    </g>
                    <g class="pl2__rect-g">
                        <rect class="pl2__rect" rx="8" ry="8" x="44" y="128" width="40" height="24" transform="rotate(180)" />
                    </g>
                    <g class="pl2__rect-g">
                        <rect class="pl2__rect" rx="8" ry="8" x="88" y="128" width="40" height="24" transform="rotate(180)" />
                    </g>
                </g>
                <g fill="hsl(283,90%,50%)" mask="url(#pl-mask)">
                    <g class="pl2__rect-g">
                        <rect class="pl2__rect" rx="8" ry="8" x="0" y="128" width="40" height="24" transform="rotate(180)" />
                    </g>
                    <g class="pl2__rect-g">
                        <rect class="pl2__rect" rx="8" ry="8" x="44" y="128" width="40" height="24" transform="rotate(180)" />
                    </g>
                    <g class="pl2__rect-g">
                        <rect class="pl2__rect" rx="8" ry="8" x="88" y="128" width="40" height="24" transform="rotate(180)" />
                    </g>
                </g>
            </svg>
            <svg class="pl3" viewBox="0 0 128 128" width="128px" height="128px">
                <g fill="var(--primary)">
                    <rect class="pl3__rect" rx="8" ry="8" width="64" height="64" transform="translate(64,0)" />
                    <g class="pl3__rect-g" transform="scale(-1,-1)">
                        <rect class="pl3__rect" rx="8" ry="8" width="64" height="64" transform="translate(64,0)" />
                    </g>
                </g>
                <g fill="hsl(163,90%,50%)" mask="url(#pl-mask)">
                    <rect class="pl3__rect" rx="8" ry="8" width="64" height="64" transform="translate(64,0)" />
                    <g class="pl3__rect-g" transform="scale(-1,-1)">
                        <rect class="pl3__rect" rx="8" ry="8" width="64" height="64" transform="translate(64,0)" />
                    </g>
                </g>
            </svg>
            
    <h2 style="font-family: Poppins, sans-serif; font-size: 2.5em; background: linear-gradient(135deg, #6a11cb, #2575fc); color: white; padding: 20px; border-radius: 15px; text-align: center;">SignLink: Bridging Communication Gaps</h2>
            <p style="font-family: Arial, sans-serif; font-size: 1.2em; text-align: center; background-color: #f0f4fc; padding: 15px; border-radius: 10px; border-bottom: 3px solid #6a11cb; color: #34495e;">Our innovative Sign Language Communication Application bridges the gap between the hearing-impaired and hearing communities, fostering smooth and inclusive interactions.</p>
            <h3 style="font-family: Arial, sans-serif; text-align: center; font-size: 2em; color: #2c3e50; margin-top: 20px;">Key Features</h3>
            <ul style="list-style-type: disc; font-family: Arial, sans-serif; color: #333; font-size: 1.1em; margin-left: 20px;">
                <li><b style="color: #6a11cb;">Sign to English with Voice:</b> Converts sign language into English text and voice for seamless communication.</li>
                <li><b style="color: #6a11cb;">Voice/Text to Sign:</b> Transforms spoken or typed words into animated sign language for accessibility.</li>
            </ul>
            <h3 style="font-family: Arial, sans-serif; text-align: center; font-size: 2em; color: #2c3e50; margin-top: 20px;">Why Choose Us?</h3>
            <ul style="list-style-type: disc; font-family: Arial, sans-serif; color: #333; font-size: 1.1em; margin-left: 20px;">
                <li><b style="color: #2575fc;">Inclusivity:</b> Promoting communication and understanding for everyone.</li>
                <li><b style="color: #2575fc;">Ease of Use:</b> Intuitive interface designed for simplicity and efficiency.</li>
                <li><b style="color: #2575fc;">Advanced Technology:</b> AI-powered translations for accuracy and reliability.</li>
            </ul>
            <h3 style="font-family: Arial, sans-serif; text-align: center; font-size: 1.8em; color: #34495e; margin-top: 20px;">Our Mission</h3>
            <p style="font-family: Arial, sans-serif; font-size: 1.2em; color: #4d4d4d; text-align: center;">To build a more inclusive society by enabling effortless communication for individuals with disabilities. Together, we can break barriers and create a world where everyone feels connected.</p></svg>
        </main>
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&display=swap'); 
            * {
                border: 0;
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                background-color: linear-gradient(#2196f3, #e91e63);
            }
            :root {
                --hue: 223;
                --bg: hsl(var(--hue),90%,90%);
                --fg: hsl(var(--hue),90%,10%);
                --primary: hsl(var(--hue),90%,50%);
                --trans-dur: 0.3s;
                font-size: calc(16px + (24 - 16) * (100vw - 320px) / (2560 - 320));
            }
            body {
                background-color: linear-gradient(#2196f3, #e91e63);
                color: var(--fg);
                display: flex;
                font: 1em/1.5 sans-serif;
                height: 100vh;
                transition:
                    background-color var(--trans-dur),
                    color var(--trans-dur);
            }
            main {
                display: flex;
                padding: 1.5em;
                gap: 3em;
                flex-wrap: wrap;
                justify-content: center;
                margin: auto;
            }
            .pl1,
            .pl2,
            .pl3 {
                display: block;
                width: 8em;
                height: 8em;
            }
            .pl1__g,
            .pl1__rect,
            .pl2__rect,
            .pl2__rect-g,
            .pl3__rect {
                animation: pl1-a 1.5s cubic-bezier(0.65,0,0.35,1) infinite;
            }
            .pl1__g {
                transform-origin: 64px 64px;
            }
            .pl1__rect:first-child {
                animation-name: pl1-b;
            }
            .pl1__rect:nth-child(2) {
                animation-name: pl1-c;
            }
            .pl2__rect,
            .pl2__rect-g {
                animation-name: pl2-a;
            }
            .pl2__rect {
                animation-name: pl2-b;
            }
            .pl2_rect-g .pl2_rect {
                transform-origin: 20px 128px;
            }
            .pl2__rect-g:first-child,
            .pl2_rect-g:first-child .pl2_rect {
                animation-delay: -0.25s;
            }
            .pl2__rect-g:nth-child(2),
            .pl2_rect-g:nth-child(2) .pl2_rect {
                animation-delay: -0.125s;
            }
            .pl2_rect-g:nth-child(2) .pl2_rect {
                transform-origin: 64px 128px;
            }
            .pl2_rect-g:nth-child(3) .pl2_rect {
                transform-origin: 108px 128px;
            }
            .pl3__rect {
                animation-name: pl3;
            }
            .pl3__rect-g {
                transform-origin: 64px 64px;
            }
            
            /* Dark theme */
            @media (prefers-color-scheme: dark) {
                :root {
                    --bg: hsl(var(--hue),90%,10%);
                    --fg: hsl(var(--hue),90%,90%);
                }
            }
            @keyframes pl1-a {
                0% { transform: scale(1); }
                25% { transform: scale(1.1); }
                50% { transform: scale(1); }
                75% { transform: scale(0.9); }
                100% { transform: scale(1); }
            }
            @keyframes pl1-b {
                0%, 25% { opacity: 1; }
                50%, 75% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            @keyframes pl1-c {
                0%, 25% { opacity: 0.5; }
                50%, 75% { opacity: 1; }
                100% { opacity: 0.5; }
            }
            @keyframes pl2-a {
                0% { transform: translateY(0); }
                50% { transform: translateY(-10px); }
                100% { transform: translateY(0); }
            }
            @keyframes pl2-b {
                0%, 25% { opacity: 1; }
                50%, 75% { opacity: 0; }
                100% { opacity: 1; }
            }
            @keyframes pl3 {
                0% { transform: translateY(0); }
                50% { transform: translateY(10px); }
                100% { transform: translateY(0); }
            }
        </style>
        """,
        unsafe_allow_html=True
    )   
elif app_mode == 'Sign Language to Text':
    st.markdown(
        """
        <h2 style="font-family: Poppins, sans-serif; font-size: 2.5em; 
        background: linear-gradient(135deg, #6a11cb, #2575fc); color: white; 
        padding: 20px; border-radius: 15px; text-align: center;">
        SignLink: Bridging Communication Gaps</h2>
        <p style="font-family: Arial, sans-serif; font-size: 1.2em; text-align: center; 
        background-color: #f0f4fc; padding: 15px; border-radius: 10px; 
        border-bottom: 3px solid #6a11cb; color: #34495e;">
        Convert sign language into text and voice effortlessly! Click the button below to start bridging the communication gap.</p>
        """,
        unsafe_allow_html=True
    )
     
    st.markdown(
        """
        <h3 style="font-family: Arial, sans-serif; text-align: center; 
        font-size: 2em; color: #2c3e50; margin-top: 20px;">Key Features</h3>
        <ul style="list-style-type: disc; font-family: Arial, sans-serif; 
        color: #333; font-size: 1.1em; margin-left: 20px;">
            <li><b style="color: #6a11cb;">Sign to English with Voice:</b> Converts sign language into English text and voice for seamless communication.</li>
            <li><b style="color: #6a11cb;">Voice/Text to Sign:</b> Transforms spoken or typed words into animated sign language for accessibility.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )
    
   # Create a button to trigger the execution of the script
    if st.sidebar.button('🚀 Click to Start Sign Language Recognition'):
        try:
            # Define the path to the script you want to run
            script_path = r"C:\Sign Language translator web\Sign-Language-To-Text-and-Speech-Conversion\final_pred.py"
            
            # Run the Python script using subprocess
            result = subprocess.run(['python', script_path], capture_output=True, text=True)
            
            # Output the result of the script (stdout, stderr) 
            st.success("Script ran successfully!")
            st.text(result.stdout)  # Display the standard output from the script
            if result.stderr:
                st.error(f"Error: {result.stderr}")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

    
else:
    st.markdown(
    """
    <h2 style="font-family: Poppins, sans-serif; font-size: 2.5em; 
    background: linear-gradient(135deg, #6a11cb, #2575fc); color: white; 
    padding: 20px; border-radius: 15px; text-align: center;">
    SignLink: Bridging Communication Gaps</h2>
    <p style="font-family: Arial, sans-serif; font-size: 1.2em; text-align: center; 
    background-color: #f0f4fc; padding: 15px; border-radius: 10px; 
    border-bottom: 3px solid #6a11cb; color: #34495e;">
    Convert text or voice into sign language seamlessly! Click the button below to start enhancing accessibility and communication.</p>
    """,
    unsafe_allow_html=True
)


    # Function to display sign language images
    def display_images(text):
        img_dir = "images/"
        image_pos = st.empty()  # Placeholder for image display

        for char in text:
            # Handle alphabets and spaces only
            if char.isalpha() or char == ' ':
                img_file = f"{char}.png" if char.isalpha() else "space.png"
                img_path = os.path.join(img_dir, img_file)

                # Check if the image file exists
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    image_pos.image(img, width=500)
                    time.sleep(1 if char.isalpha() else 2)  # Timing for display
                    image_pos.empty()
                else:
                    st.warning(f"Image for '{char}' not found.")
            else:
                st.warning(f"Unsupported character: '{char}'")
        
        time.sleep(2)  # Pause after final image
        image_pos.empty()


    # Function for speech recognition
    def speech_to_text():
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                st.info("Listening... Please speak.")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text.upper()  # Convert to uppercase for uniformity
        except sr.UnknownValueError:
            st.error("Sorry, I couldn't understand what you said.")
        except sr.RequestError as e:
            st.error(f"Error with the speech recognition service: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
        return ""


    # Input and action buttons
    col1, col2 = st.columns([3, 1])

    with col1:
        # Text input box
        text = st.text_input("Enter text:")

    with col2:
        # Speech-to-text button
        if st.button("Speak"):
            recognized_text = speech_to_text()
            if recognized_text:
                text = recognized_text  # Update text box with recognized speech

    # Convert text to uppercase for processing
    text = text.upper()

    # Display sign language images if text is provided
    if text:
        display_images(text)