import os
import subprocess
import sys

# Function to execute a command and handle errors
def execute_command(command, step_description):
    print(f"Step: {step_description}")
    print(f"Executing: {command}")
    try:
        subprocess.check_call(command, shell=True)
        print(f"✓ {step_description} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in: {step_description}.")
        print(f"Details: {e}")
        print("Exiting setup to resolve the issue.")
        sys.exit(1)

# Confirmations
print("Welcome to the setup script for your project.")
print("Ensure you are running this script in the correct environment (e.g., a virtual environment).")
input("Press Enter to continue...")

# Step 1: Downgrade NumPy
execute_command("pip install numpy==1.23.5", "Downgrade NumPy to version 1.23.5")

# Step 2: Install Keras
execute_command("pip install keras", "Install Keras")

# Step 3: Install cvzone and mediapipe
execute_command("pip install cvzone mediapipe", "Install cvzone and mediapipe")

# Step 4: Install pyttsx3
execute_command("pip install pyttsx3", "Install pyttsx3 (text-to-speech)")

# Step 5: Install Pillow
execute_command("pip install pillow", "Install Pillow (image processing)")

# Step 6: Install OpenCV
execute_command("pip install opencv-python-headless", "Install OpenCV (cv2)")

# Step 7: Install PyEnchant
execute_command("pip install pyenchant", "Install PyEnchant (spell-checking)")

# Step 8: Install Theano
execute_command("pip install Theano", "Install Theano (optional for Theano flags)")

# Additional confirmations
print("\nSetup commands have been executed.")
print("Ensure the following files are in place:")
print("1. `cnn8grps_rad1_model.h5` (model file)")
print("2. `white.jpg` (referenced in the script)")
input("Press Enter to confirm these files exist and continue...")

# Final Message
print("\nSetup complete!")
print("You can now run your project script. If any issues persist, check the logs above and reinstall specific modules as needed.")
