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
            <svg class="pl1" viewBox="0 0 128 128" width="128px" height="128px">
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
            <h2 style="text-align: center; margin-top: 20px;">Sign Language APP</h2>
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
    st.title('Sign Language to Text')
    #st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")
    if record:
        st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    sameer=""
    st.markdown(' ## Output')
    st.markdown(sameer)

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO

    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4
    while True:
        ret, img = vid.read()
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                lm_list = []
                for id, lm in enumerate(hand_landmark.landmark):
                    lm_list.append(lm)
                finger_fold_status = []
                for tip in finger_tips:
                    x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)
                    # print(id, ":", x, y)
                    # cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)

                    if lm_list[tip].x < lm_list[tip - 2].x:
                        # cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)
                        finger_fold_status.append(True)
                    else:
                        finger_fold_status.append(False)

                print(finger_fold_status)
                x, y = int(lm_list[8].x * w), int(lm_list[8].y * h)
                print(x, y)
                # fuck off
                #if lm_list[3].x < lm_list[4].x and lm_list[8].y > lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                   #     lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                 #   cv2.putText(img, "fuck off !!!", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                   # sameer="fuck off"

                # one
                if lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[4].y < lm_list[
                    12].y:
                    cv2.putText(img, "1", (170, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("1")

                # two
                if lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                    cv2.putText(img, "2", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("2")
                    sameer="2"
                # three
                if lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                    cv2.putText(img, "3", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("3")
                    sameer="3"

                # four
                if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[2].x < lm_list[8].x:
                    cv2.putText(img, "4", (170, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("4")
                    sameer="4"

                # five
                if lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x:
                    cv2.putText(img, "5", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("5") 
                    sameer="5"
                    # six
                if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x:
                    cv2.putText(img, "6", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("6")
                    sameer="6"
                # SEVEN
                if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x:
                    cv2.putText(img, "7", (70, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("7")
                    sameer="7"
                elif all([
                    lm_list[4].x < lm_list[3].x,  # Thumb is extended outward
                    lm_list[8].y < lm_list[6].y,  # Index finger is extended upwards
                    lm_list[12].y < lm_list[10].y,  # Middle finger is extended upwards
                    lm_list[16].y > lm_list[14].y,  # Ring finger is bent downwards
                    lm_list[20].y > lm_list[18].y  # Pinky finger is bent downwards
                ]):
                    cv2.putText(img, "7", (70, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("7")
                    sameer = "7"
                # EIGHT
                if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x:
                    cv2.putText(img, "8", (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("8")
                    sameer="8"
                # NINE
                if lm_list[2].x > lm_list[4].x and lm_list[8].y > lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x:
                    cv2.putText(img, "9", (170, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("9")
                    sameer="9"
                # A
                if lm_list[2].y > lm_list[4].y and lm_list[8].y > lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x and lm_list[4].y < lm_list[6].y:
                    cv2.putText(img, "A", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("A")
                # B
                if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[2].x > lm_list[8].x:
                    cv2.putText(img, "B", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("B")
                    sameer="B"
                # c
                if lm_list[2].x < lm_list[4].x and lm_list[8].x > lm_list[6].x and lm_list[12].x > lm_list[10].x and \
                        lm_list[16].x > lm_list[14].x and lm_list[20].x > lm_list[18].x:
                    cv2.putText(img, "C", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("C")
                # d
                if lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[4].y > lm_list[8].y:
                    cv2.putText(img, "D", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("D")

                # E
                if lm_list[2].x > lm_list[4].x and lm_list[8].y > lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x and lm_list[4].y > lm_list[6].y:
                    cv2.putText(img, "E", (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("E")
                # X
                elif all([lm_list[4].x > lm_list[3].x, lm_list[8].y < lm_list[6].y, lm_list[12].y > lm_list[10].y,
                    lm_list[16].y > lm_list[14].y, lm_list[20].y > lm_list[18].y]):
                    cv2.putText(img, "X", (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("X")
                    sameer = "X"

                # G
# Detecting the letter G
                if all([
                    lm_list[4].x > lm_list[3].x,    # Thumb extended horizontally (landmark 4 to the right of 3)
                    lm_list[8].x > lm_list[6].x,    # Index finger extended horizontally (landmark 8 to the right of 6)
                    lm_list[12].y > lm_list[10].y,  # Middle finger closed
                    lm_list[16].y > lm_list[14].y,  # Ring finger closed
                    lm_list[20].y > lm_list[18].y   # Pinky closed
                ]):
                    cv2.putText(img, "G", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("G")
                    sameer = "G"


                # H
                # Detecting the letter H
                if all([
                    lm_list[8].x > lm_list[6].x,    # Index finger extended horizontally
                    lm_list[12].x > lm_list[10].x,  # Middle finger extended horizontally
                    lm_list[16].y > lm_list[14].y,  # Ring finger closed
                    lm_list[20].y > lm_list[18].y,  # Pinky closed
                    lm_list[4].x < lm_list[3].x     # Thumb closed or bent
                ]):
                    cv2.putText(img, "H", (70, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("H")
                    sameer = "H"


                # I
                # Detecting the letter I
                if all([
                    lm_list[20].y < lm_list[19].y,  # Pinky extended
                    lm_list[8].y > lm_list[6].y,    # Index finger closed
                    lm_list[12].y > lm_list[10].y,  # Middle finger closed
                    lm_list[16].y > lm_list[14].y,  # Ring finger closed
                    lm_list[4].x < lm_list[3].x     # Thumb closed or bent
                ]):
                    cv2.putText(img, "I", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("I")
                    sameer = "I"


                # Variable to store previous positions of the pinky fingertip (landmark 20)
                threshold=20
                previous_pinky_positions = []

                # Function to detect the "J" arc movement
                def detect_j_movement(positions):
                    if len(positions) < 3:
                        return False

                    # Define three segments: starting position, curve down, and finishing at a horizontal-like movement
                    start_segment = positions[-3]
                    middle_segment = positions[-2]
                    end_segment = positions[-1]

                    # Check for downward curved motion
                    curve_down = (middle_segment[1] > start_segment[1])  # Pinky moves downward

                    # Check for left or right curve (horizontal arc)
                    curve_horizontal = abs(middle_segment[0] - start_segment[0]) > threshold

                    # Final slight upward or horizontal motion to close the J shape
                    end_motion = (end_segment[1] < middle_segment[1]) or (abs(end_segment[0] - middle_segment[0]) > threshold)

                    # If all conditions are satisfied, we have detected a "J" shape
                    return curve_down and curve_horizontal and end_motion

                # Add logic to track pinky fingertip positions (lm_list[20]) across frames
                current_pinky_position = (lm_list[20].x, lm_list[20].y)
                previous_pinky_positions.append(current_pinky_position)

                # Limit the number of stored positions to the last 3 to 5 frames
                if len(previous_pinky_positions) > 5:
                    previous_pinky_positions.pop(0)

                # Check if the current gesture looks like a "J"
                if detect_j_movement(previous_pinky_positions):
                    cv2.putText(img, "J", (70, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("J")
                    sameer = "J"


                # K
                # Detecting the letter K
                if all([
                    lm_list[8].y < lm_list[6].y,    # Index finger extended
                    lm_list[12].y < lm_list[10].y,  # Middle finger extended
                    lm_list[4].x < lm_list[3].x,    # Thumb extended upward
                    lm_list[16].y > lm_list[14].y,  # Ring finger closed
                    lm_list[20].y > lm_list[18].y   # Pinky closed
                ]):
                    cv2.putText(img, "K", (70, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("K")
                    sameer = "K"


                # L
                # Detecting the letter L
                if all([
                    lm_list[8].y < lm_list[6].y,    # Index finger extended
                    lm_list[4].x > lm_list[3].x,    # Thumb extended forming the L shape
                    lm_list[12].y > lm_list[10].y,  # Middle finger closed
                    lm_list[16].y > lm_list[14].y,  # Ring finger closed
                    lm_list[20].y > lm_list[18].y   # Pinky closed
                ]):
                    cv2.putText(img, "L", (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("L")
                    sameer = "L" 


                # M
                # Detecting the letter M
                if all([
                    lm_list[4].x > lm_list[3].x,  # Thumb is folded across the fingers
                    lm_list[8].y > lm_list[6].y,  # Index finger is folded downwards
                    lm_list[12].y > lm_list[10].y,  # Middle finger is folded downwards
                    lm_list[16].y > lm_list[14].y,  # Ring finger is folded downwards
                    lm_list[20].y > lm_list[18].y  # Pinky finger is folded downwards
                ]):
                    cv2.putText(img, "M", (70, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("M")
                    sameer = "M"


                # N
                # Detecting the letter N
                if all([
                    lm_list[8].y > lm_list[6].y,    # Index finger folded over thumb
                    lm_list[12].y > lm_list[10].y,  # Middle finger folded over thumb
                    lm_list[16].y < lm_list[14].y,  # Ring finger extended
                    lm_list[20].y < lm_list[19].y   # Pinky extended
                ]):
                    cv2.putText(img, "N", (170, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("N")
                    sameer = "N"


                # O
                elif all([lm_list[4].x > lm_list[3].x, lm_list[8].y > lm_list[6].y, lm_list[12].y > lm_list[10].y,
                        lm_list[16].y > lm_list[14].y, lm_list[20].y > lm_list[18].y]):
                    cv2.putText(img, "O", (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("O")
                    sameer = "O"

                # P
                # Detecting the letter P
                if all([
                    lm_list[8].y > lm_list[6].y,    # Index finger pointing downward
                    lm_list[12].y > lm_list[10].y,  # Middle finger pointing downward
                    lm_list[4].x < lm_list[3].x,    # Thumb extended outward
                    lm_list[16].y > lm_list[14].y,  # Ring finger closed
                    lm_list[20].y > lm_list[18].y   # Pinky closed
                ]):
                    cv2.putText(img, "P", (70, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("P")
                    sameer = "P"


                # Q
                elif all([lm_list[4].x < lm_list[3].x, lm_list[8].x < lm_list[6].x, lm_list[12].y > lm_list[10].y]):
                    cv2.putText(img, "Q", (70, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("Q")
                    sameer = "Q"

                # R
                # Detecting the letter R
                if all([
                    lm_list[8].x < lm_list[12].x,   # Index and middle fingers crossed
                    lm_list[16].y > lm_list[14].y,  # Ring finger closed
                    lm_list[20].y > lm_list[18].y,  # Pinky closed
                    lm_list[4].x < lm_list[3].x     # Thumb closed or bent
                ]):
                    cv2.putText(img, "R", (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("R")
                    sameer = "R"


                # S
                # Detecting the letter S
                # Detecting the letter S
                elif all([
                    lm_list[4].y < lm_list[3].y,  # Thumb is slightly up, not tucked under the palm
                    lm_list[8].y > lm_list[7].y,  # Index finger is curled into the palm
                    lm_list[12].y < lm_list[11].y,  # Middle finger is slightly higher than the others
                    lm_list[16].y > lm_list[15].y,  # Ring finger is curled into the palm
                    lm_list[20].y > lm_list[19].y  # Pinky finger is curled into the palm
                ]):
                    cv2.putText(img, "S", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("S")
                    sameer = "S"


                # T
                # Detecting the letter T
                if all([
                    lm_list[4].x > lm_list[3].x,    # Thumb tucked under index
                    lm_list[8].y > lm_list[6].y,    # Index finger closed over thumb
                    lm_list[12].y > lm_list[10].y,  # Middle finger closed
                    lm_list[16].y > lm_list[14].y,  # Ring finger closed
                    lm_list[20].y > lm_list[18].y   # Pinky closed
                ]):
                    cv2.putText(img, "T", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("T")
                    sameer = "T"


                # U
                # Detecting the letter U
                if all([
                    lm_list[8].y < lm_list[6].y,    # Index finger extended
                    lm_list[12].y < lm_list[10].y,  # Middle finger extended
                    lm_list[16].y > lm_list[14].y,  # Ring finger closed
                    lm_list[20].y > lm_list[18].y,  # Pinky closed
                    lm_list[4].x < lm_list[3].x     # Thumb closed or bent
                ]):
                    cv2.putText(img, "U", (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("U")
                    sameer = "U"


                # V
                # Detecting the letter V
                if all([
                    lm_list[8].y < lm_list[6].y,    # Index finger extended
                    lm_list[12].y < lm_list[10].y,  # Middle finger extended
                    lm_list[16].y > lm_list[14].y,  # Ring finger closed
                    lm_list[20].y > lm_list[18].y,  # Pinky closed
                    abs(lm_list[8].x - lm_list[12].x) > threshold  # Index and middle fingers separated
                ]):
                    cv2.putText(img, "V", (170, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("V")
                    sameer = "V"


                # W
                # Detecting the letter W
                if all([
                    lm_list[8].y < lm_list[6].y,    # Index finger extended
                    lm_list[12].y < lm_list[10].y,  # Middle finger extended
                    lm_list[16].y < lm_list[14].y,  # Ring finger extended
                    lm_list[20].y > lm_list[18].y,  # Pinky closed
                    lm_list[4].x < lm_list[3].x     # Thumb closed or bent
                ]):
                    cv2.putText(img, "W", (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("W")
                    sameer = "W"


                # F
                elif all([lm_list[4].y < lm_list[3].y, lm_list[8].y > lm_list[6].y, lm_list[12].y < lm_list[10].y]):
                    cv2.putText(img, "F", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("F")
                    sameer = "F"

                # Y
                # Detecting the letter Y
                if all([
                    lm_list[4].y < lm_list[3].y,    # Thumb extended (tip above joint)
                    lm_list[20].y < lm_list[19].y,  # Pinky extended (tip above joint)
                    lm_list[8].y > lm_list[6].y,    # Index finger closed (tip below joint)
                    lm_list[12].y > lm_list[10].y,  # Middle finger closed (tip below joint)
                    lm_list[16].y > lm_list[14].y   # Ring finger closed (tip below joint)
                ]):
                    cv2.putText(img, "Y", (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("Y")
                    sameer = "Y"
                    


                # Z
                # Detecting the letter Z (movement involved)
                # Variable to store previous positions of the index fingertip (landmark 8)
                previous_positions = []

                # Function to detect the movement pattern of "Z"
                def detect_z_movement(positions):
                    if len(positions) < 3:
                        return False

                    # Check for three segments: diagonal down, horizontal, diagonal up
                    first_segment = positions[-3]
                    second_segment = positions[-2]
                    third_segment = positions[-1]

                    # Segment 1: Check diagonal movement (e.g., down-left or down-right)
                    diagonal1 = (third_segment[1] > second_segment[1]) and (third_segment[0] != second_segment[0])

                    # Segment 2: Check horizontal movement
                    horizontal = abs(third_segment[1] - second_segment[1]) < threshold and abs(third_segment[0] - second_segment[0]) > threshold

                    # Segment 3: Check diagonal movement (up-left or up-right)
                    diagonal2 = (third_segment[1] < second_segment[1]) and (third_segment[0] != second_segment[0])

                    # If all conditions are satisfied, we detected a "Z"
                    return diagonal1 and horizontal and diagonal2

                # Add logic to track index fingertip positions (lm_list[8]) across frames
                current_position = (lm_list[8].x, lm_list[8].y)
                previous_positions.append(current_position)

                # Limit the number of stored positions to the last 3 to 5 frames
                if len(previous_positions) > 5:
                    previous_positions.pop(0)

                # Check if the current gesture looks like a "Z"
                if detect_z_movement(previous_positions):
                    cv2.putText(img, "Z", (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("Z")
                    sameer = "Z"


            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec((0, 0, 255), 6, 3),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2)
                                   )
            if record:

                out.write(img)


            frame = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
            frame = image_resize(image=frame, width=640)
            stframe.image(frame, channels='BGR', use_column_width=True)

    st.text('Video Processed')

    output_video = open('output1.mp4', 'rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    out.release()
else:
    st.title('Text to Sign Language')


   # Define function to display sign language images
    def display_images(text):
        # Get the file path of the images directory
        img_dir = "images/"

        # Initialize variable to track image position
        image_pos = st.empty()

        # Iterate through the text and display sign language images
        for char in text:
            if char.isalpha():
                # Display sign language image for the alphabet
                img_path = os.path.join(img_dir, f"{char}.png")
                img = Image.open(img_path)

                # Update the position of the image
                image_pos.image(img, width=500)

                # Wait for 1 second before displaying the next image
                time.sleep(1)

                # Remove the image
                image_pos.empty()
            elif char == ' ':
                # Display space image for space character
                img_path = os.path.join(img_dir, "space.png")
                img = Image.open(img_path)

                # Update the position of the image
                image_pos.image(img, width=500)

                # Wait for 2 seconds before displaying the next image
                time.sleep(2)

                # Remove the image
                image_pos.empty()

        # Wait for 2 seconds before removing the last image
        time.sleep(2)
        image_pos.empty()

    # Define function for speech recognition
    def speech_to_text():
        # Initialize recognizer
        recognizer = sr.Recognizer()

        try:
            with sr.Microphone() as source:
                st.info("Listening... Please speak.")
                recognizer.adjust_for_ambient_noise(source)
                # Adjust the following parameters as needed
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            # Using Google Web Speech API to recognize speech
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text.upper()  # Convert text to uppercase
        except sr.UnknownValueError:
            st.error("Sorry, I couldn't understand what you said.")
        except sr.RequestError as e:
            st.error(f"Error requesting results from Google Speech Recognition service: {e}")
        except Exception as e:
            st.error(f"Error: {e}")
        return ""

    # Main Streamlit app
    st.title('Text to Sign Language')

    # Create two columns for the text box and button
    col1, col2 = st.columns([3, 1])

    # Text input in the first column
    with col1:
        text = st.text_input("Enter text:")

    # Button for speech recognition in the second column
    with col2:
        if st.button("Speak"):
            # Run speech recognition and get the recognized text
            recognized_text = speech_to_text()
            # If speech was recognized, put it into the text box
            if recognized_text:
                text = recognized_text

    # Convert text to uppercase
    text = text.upper()

    # Display sign language images
    if text:
        display_images(text)