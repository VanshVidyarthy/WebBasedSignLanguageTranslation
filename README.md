Sign Language Detection System
A real-time Sign Language Detection System that converts hand gestures to English text and English text to sign language. This project leverages MediaPipe, Streamlit, and advanced machine learning techniques to enable communication between sign language users and non-sign language users.

Features
Sign Language to English Conversion:
Captures hand gestures using a webcam and converts them to corresponding English text in real-time using MediaPipe's hand landmark detection.

English to Sign Language Conversion:
Converts English alphabets or words entered by the user into corresponding sign language images.

Intuitive User Interface:
Built using Streamlit, the application offers an easy-to-use interface with a sidebar for navigation.

Technology Stack
Programming Language: Python
Libraries and Frameworks:
Streamlit for UI development.
MediaPipe for real-time hand gesture detection.
OpenCV for image processing.
NumPy for efficient data handling.
PIL (Pillow) for image manipulation.
SpeechRecognition for additional features like voice input (optional).
Installation and Setup
Prerequisites
Python 3.7 or later
Libraries listed in requirements.txt

Installation Steps
1.Clone the repository:
git clone https://github.com/yourusername/sign-language-detection.git
cd sign-language-detection
2.Install the required libraries:
pip install -r requirements.txt
3.Run the application:
streamlit run signlanguagedetection.py

How to Use
Launch the application using the command mentioned above. Paste the command streamlit run signlanguagedetection.py in powershell or using command prompt in the directory in which the python file exists.
Navigate through the sidebar to select:
About App: Learn more about the project.
Sign Language to Text: Use your webcam to translate gestures into English text.
Text to Sign Language: Input English text to see corresponding sign language images.
Interact with the application and explore both features in real-time.

Images Folder Details
The images folder contains 26 PNG files, each representing a sign language gesture for the English alphabets (A-Z). These are preloaded in the application for the English to Sign Language Conversion feature.

License
This project is licensed under the MIT License.
