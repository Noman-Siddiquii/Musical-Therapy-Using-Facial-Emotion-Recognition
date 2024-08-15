import streamlit as st
import cv2
import numpy as np
import threading
from tensorflow.keras.models import load_model
import pygame

try:
  model = load_model('Final_model.h5')
except FileNotFoundError:
  st.error("Model file not found. Please check the path and filename.")

# Function to preprocess the image
def preprocess_image(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  resized = cv2.resize(gray, (48, 48))
  normalized = resized / 255.0
  reshaped = np.reshape(normalized, (1, 48, 48, 1))
  return reshaped

# Function to predict the emotion
def predict_emotion(img):
    try:
        preprocessed_img = preprocess_image(img)
        prediction = model.predict(preprocessed_img)
        st.write("Prediction Raw Output:", prediction)
        emotion = np.argmax(prediction)
        st.write("Predicted Emotion Index:", emotion)
        return emotion
    except Exception as e:
        st.error(f"Error predicting emotion: {e}")
        return None


# Function to play music based on emotion
def play_music(emotion):
  def play_music_thread():
    if emotion == 0:  # Happy
      pygame.mixer.music.load('happy.mp3')
    elif emotion == 1:  # sad
      pygame.mixer.music.load('sad.mp3')
    try:
      pygame.mixer.music.play()
    except pygame.error as e:
      st.error("Error playing music:", e)

  # Create and start a new thread
  thread = threading.Thread(target=play_music_thread)
  thread.start()

# Initialize Pygame mixer
pygame.mixer.init()

def main():
  """Streamlit application for facial expression recognition and music playback"""
  st.title("Musical Therapy with Facial Expressions")

  # Option for webcam or upload
  source = st.selectbox("Choose Source", ["Webcam", "Upload Image"])

  # Webcam feed
  if source == "Webcam":
    run_webcam = st.checkbox("Run Webcam")
    cap = cv2.VideoCapture(0)

    while run_webcam:
      ret, frame = cap.read()

      if not ret:
        st.error("Error capturing frame from webcam.")
        break

      # Process and analyze webcam frame
      #

      # Exit on 'q' key press
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()

  # Image upload
  elif source == "Upload Image":
    uploaded_file = st.file_uploader("Choose an Image", type="jpg")
    if uploaded_file is not None:
      image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
      emotion = predict_emotion(image)
      if emotion == 0:
        st.write('The Facial Expressions are Happy')
      elif emotion == 1:
        st.write('The Facial Expressions are sad')
      play_music(emotion)

      st.write('The Music is playing....')

      # Process and analyze uploaded image
      

  # Release resources and stop music
  pygame.mixer.music.stop()

if __name__ == "__main__":
  main()
