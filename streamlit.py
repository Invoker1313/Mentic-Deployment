import streamlit as st
import spacy
from sklearn.tree import DecisionTreeClassifier
import speech_recognition as sr
import json
from datetime import datetime
from spacy.cli import download

# Download the en_core_web_md model
#download("en_core_web_md")
# Load spaCy model

#nlp = spacy.load('en_core_web_md')
# Ensure the model is downloaded

#global depressed_mood_counter, anhedonia_counter
#global appetite_counter, insomnia_counter
#global worthlessness_counter, emotion_counter, negative_counter
#global cognitive_counter 
#global suicidal_counter
#global fatigue_counter

# Define counters
depressed_mood_counter = anhedonia_counter = appetite_counter = insomnia_counter = fatigue_counter = 0
worthlessness_counter = emotion_counter = negative_counter = cognitive_counter = suicidal_counter = 0

# Define lists of keywords for symptoms (already defined in your initial code)
# General Terms Indicating Depression
depressed_mood = [
    "sad", "unhappiness", "sorrow", "down", "gloom", "despair", "sorrowful", 
    "hopeless", "miserable", "lonely", "isolated", "alone", "empty", "abandoned", 
    "distant", "worthless", "inadequate", "useless", "failure", "incompetent", "meaningless"
]

# Lack of Motivation
lack_of_motivation = [
    "unmotivated", "tired", "exhausted", "drained", "apathetic", "numb", 
    "burnt out", "indifferent"
]

# Negative Self-view
negative_self_view = [
    "hate me", "loath myself", "hate myself", "inferior", "doubt", "disgust", "disgusted"
]

# Emotional Expressions
emotional_expressions = [
    "anxious", "nervous", "afraid", "scared", "uneasy", "panicked", "fearful", 
    "hopeless", "no hope", "discouraged", "can't go on", "helpless", "lost cause", 
    "crying", "tears", "sobbing", "broken", "heartbroken", "crushed", "resigned", 
    "given up", "fed up", "defeated", "ready to quit", "nothing matters"
]

# Physical and Mental Fatigue
physical_mental_fatigue = [
    "tired", "exhausted", "weary", "drained", "fatigued", "lifeless", 
    "no energy", "can’t sleep", "insomnia", "tired all the time", "oversleeping", 
    "broken sleep", "headaches", "body aches", "weight", "heaviness", "sore", 
    "stiff", "can’t move"
]

# Cognitive Distortions
cognitive_distortions = [
    "can’t stop thinking", "replaying thoughts", "overthinking", "obsessing", 
    "stuck in my head", "can’t decide", "unsure", "confused", "lost", 
    "don’t know what to do", "can’t focus", "distracted", "can’t concentrate", 
    "foggy", "brain fog"
]

# Loss of Interest and Pleasure
loss_of_interest_pleasure_anhedonia = [
    "no interest", "don’t care", "lost passion", "meaningless", "pointless", 
    "can’t enjoy", "withdrawing", "shutting out", "avoiding", "stopped doing", 
    "isolating"
]

# Expressing Worthlessness and Hopelessness
worthlessness_hopelessness = [
    "no future", "nothing to live for", "no reason", "pointless", "meaningless", 
    "empty inside", "hate myself", "always failing", "useless", "disappointed in myself", 
    "nothing I do matters", "life is awful", "everything is bad", "always happens to me", 
    "what’s the point"
]

# Suicidal Thoughts (Important for Risk Detection)
suicidal_thoughts = [
    "want to die", "suicidal", "end it all", "can’t keep going", "don’t want to live", 
    "better off dead", "world would be better", "want to disappear", "escape everything", 
    "wish I wasn’t here"
]

# Physical Complaints
change_appetite = [
    "chronic pain", "stomach issues", "headaches", "low energy", "physical discomfort", 
    "no appetite", "overeating", "eating too little", "weight loss", "weight gain"
]

# Cognitive Bias and Negative View of the World
cognitive_bias_negative_view = [
    "life is unfair", "no one cares", "can’t trust anyone", "everyone’s out to get me", 
    "futureless", "bleak", "won’t get better", "never-ending pain"
]

#Negation Handling
negation_handler = [
    "not", "none", "never"
]

self_subject_check = [
    "I", "me", "mine", "myself"
]

# Define functions for dependency parsing, negation, etc. (from initial code)
def dependency_parsing(user_speech):
    nlp = spacy.load('en_core_web_md')
    doc = nlp(user_speech)

    subject = None

    for token in doc:
        if token.dep_ == "nsubj":
            for pointer in self_subject_check:
                if token == pointer:
                    return True
                
    else:
        return False
    
def negation_check(word):
    for check in negation_handler:
        if check in word:
            return True
        
def lemmatizer(user_speech):
    doc = nlp(user_speech)
    for token in doc:
        lemma_doc = ' ' + token.lemma_
    return lemma_doc

def depressedmood_check(user_speech):
    global depressed_mood_counter
    for token in depressed_mood:
        if token in user_speech:
            #if dependency_parsing(user_speech) == True:
            depressed_mood_counter = depressed_mood_counter + 1
        return depressed_mood_counter
    
def anhedonia(user_speech):
    global anhedonia_counter
    for token in loss_of_interest_pleasure_anhedonia:
        if token in user_speech:
            #if dependency_parsing(user_speech) == True:
                anhedonia_counter = anhedonia_counter + 1
        return anhedonia_counter
    
def negative_self_view_check(user_speech):
    global negative_counter
    for token in negative_self_view:
        if token in user_speech:
            negative_counter = negative_counter + 1
        return negative_counter
        
def emotional_expression_check(user_speech):
    global emotion_counter
    for token in emotional_expressions_check:
        if token in user_speech:
            emotion_counter = emotion_counter + 1
            return emotion_counter
        
def fatigue_chceck(user_speech):
    global fatigue_counter
    for token in physical_mental_fatigue:
        if token in user_speech:
            #if dependency_parsing(user_speech) == True:
                fatigue_counter = fatigue_counter + 1
        return fatigue_counter
        
def guilt_check(user_speech):
    global guilt_counter
    for token in worthlessness_hopelessness:
        if token in user_speech:
            #if dependency_parsing(user_speech) == True:
                guilt_counter = guilt_counter + 1
        return guilt_counter

def cognitive_check(user_speech):
    global cognitive_counter
    for token in cognitive_distortions:
        if token in user_speech:
            #if dependency_parsing(user_speech) == True:
                cognitive_counter = cognitive_counter + 1
        return cognitive_counter

def suicidal_check(user_speech):
    global suicidal_counter
    for token in suicidal_thoughts:
        if token in user_speech:
            #if dependency_parsing(user_speech) == True:
                suicidal_counter = suicidal_counter + 1
        return suicidal_counter
        
def get_audio_input():
    with sr.Microphone() as source:
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.error("Could not understand the audio")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
    return None

# Define each symptom check function (from initial code)

import streamlit as st
import joblib
from vosk import Model, KaldiRecognizer
#import pyaudio
from groq import Groq  # Hypothetical GroqLLM package
GroqAPIKey = "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"
GroqClient = Groq(api_key = GroqAPIKey)
GroqModel = "llama3-70b-8192"

# Load pretrained classifier
clf = joblib.load('classifier_model.joblib')  # Replace with your model path

# Initialize conversation context in session state
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = []  # Storing context (user inputs + LLM responses)
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False  # To manage microphone recording
if "user_texts" not in st.session_state: # This for analysis later on
    st.session_state.user_texts = ""
if "analysis" not in st.session_state:
    st.session_state.analysis = []

# Initialize Vosk STT Model (Ensure the model is downloaded)
vosk_model = Model("C:/Users/adity/ISWAD/AI Job Interview APP/shared/storage/STT Vosk Models/vosk-model-small-en-us-0.15")  # Path to your Vosk STT model directory
#recognizer = KaldiRecognizer(vosk_model, 16000)
recognizer = sr.Recognizer()

# Streamlit UI
st.title("Depression Severity Checker with Conversational LLM")

# Button to start/stop recording
if st.session_state.is_recording:
    if st.button("Stop Recording"):
        st.session_state.is_recording = False
else:
    if st.button("Start Recording"):
        st.session_state.is_recording = True

# Recording Functionality
def record_audio():
    """Record audio from the microphone and convert it to text using Vosk."""
    if st.session_state.is_recording:
        st.info("Recording... Speak into the microphone.")
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        stream.start_stream()
        text = ""

        while st.session_state.is_recording:
            data = stream.read(4000, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                text += result

        stream.stop_stream()
        stream.close()
        audio.terminate()
        return text

# Set a predefined username and password (you can use environment variables for production)
USERNAME = "admin"
PASSWORD = "password123"

# Initialize authentication state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Login Form
if not st.session_state.authenticated:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.authenticated = True
            st.success("Login successful!")
        else:
            st.error("Invalid username or password")
else:
    # Main app logic after authentication
    st.title("Welcome to the Secure App")
    st.write("You are now logged in!")
    
    # Logout button
    if st.button("Logout"):
        st.session_state.authenticated = False


    # Use the microphone to get user input
    if st.session_state.is_recording:
        user_text = get_audio_input()
        if user_text:
            st.success(f"You Said: {user_text}")
            st.session_state.conversation_context.append({"role": "user", "content": user_text})
            st.session_state.user_texts = st.session_state.user_texts + " " + user_text

    # Send user input to GroqLLM for a response
    if st.button("Get LLM Response"):
        if st.session_state.conversation_context:
            user_input = st.session_state.conversation_context
            response = GroqClient.chat.completions.create(
                model = GroqModel,
                messages = user_input
            )
            st.session_state.conversation_context.append({"role": "assistant", "content": response.choices[0].message.content})
            st.success(f"LLM Response: {response.choices[0].message.content}")

    # Display the conversation context
    st.write("**Conversation Context:**")
    for exchange in st.session_state.conversation_context:
        for role, text in exchange.items():
            st.write(f"**{role}:** {text}")

    # Questions for appetite and sleep cycle changes
    st.write("**Additional Questions:**")
    appetite_change = st.radio("Did you experience any change in diet?", ("Yes", "No"))
    insomnia_change = st.radio("Have you had any change in your sleep cycle?", ("Yes", "No"))

    # Convert answers to binary format
    appetite_counter = 1 if appetite_change == "Yes" else 0
    insomnia_counter = 1 if insomnia_change == "Yes" else 0

    # Button to analyze the conversation
    if st.button("Analyze"):
        # Combine all user inputs from the conversation
        #combined_input = " ".join(
        #    [exchange["user"] for exchange in st.session_state.conversation_context if "user" in exchange]
        #)
        combined_input = st.session_state.user_texts
        st.write(st.session_state.conversation_context)

        st.write("User inputs - ", combined_input)

        # Reset counters for feature extraction
        depressed_mood_counter = 0
        anhedonia_counter = 0
        #fatigue_counter = 0
        worthlessness_counter = 0
        emotion_counter = 0
        negative_counter = 0
        cognitive_counter = 0
        suicidal_counter = 0
        guilt_counter = 0

        # Example: Run checks and populate counters based on combined input
        depressed_mood_counter = depressedmood_check(combined_input)
        anhedonia_counter = anhedonia(combined_input)
        negative_counter = negative_self_view_check(combined_input)
        fatigue_counter = fatigue_chceck(combined_input)
        worthlessness_counter = guilt_check(combined_input)

        # Compile data in the specified format
        sample_data = [[depressed_mood_counter, anhedonia_counter, appetite_counter, insomnia_counter,
                        fatigue_counter, worthlessness_counter, emotion_counter, negative_counter,
                        cognitive_counter, suicidal_counter]]

        # Display the extracted data
        st.write("**Extracted Symptoms Data:**", sample_data)

        # Use pretrained model for prediction
        prediction = clf.predict(sample_data)
        st.write("**Predicted Depression Severity Level:**", prediction[0])

        st.session_state.analysis = {
            "Depressed Mood" : depressed_mood_counter,
            "Anhedonia" : anhedonia_counter,
            "Appetite Change" : appetite_counter,
            "Insomnia" : insomnia_counter,
            "Fatigue" : fatigue_counter,
            "Feeling of Worthlessness" : worthlessness_counter,
            "Emotional References" : emotion_counter,
            "Negative Self View" : negative_counter,
            "Cognitive" : cognitive_counter,
            "Suicidal Thoughts" : suicidal_counter
        }

        if st.session_state.conversation_context and st.session_state.analysis:
            report = {
                "timestamp": datetime.now().isoformat(),
                "conversation": st.session_state.conversation_context,
                "analysis": st.session_state.analysis
            }
            # Save to a JSON file
            with open("user_report.json", "a") as f:
                f.write(json.dumps(report) + "\n")
            st.success("Report saved successfully!")
        else:
            st.warning("No data to save!")
