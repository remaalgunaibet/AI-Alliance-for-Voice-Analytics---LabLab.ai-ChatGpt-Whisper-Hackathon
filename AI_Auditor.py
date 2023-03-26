import gradio as gr
import openai
import config
from gtts import gTTS
import os
import subprocess
from pydub import AudioSegment
import math

openai.api_key = "YOUR_API_KEY"
messages = [
    {"role": "system", "content": "You are a call center quality and assurance auditor. Your job is to review the call recording, and provide a very brief summary of the key information in the call including Operator’s Name, Call Category, Issue, and Solution. Also, you need to conduct sentiment analysis on the call and evaluate the customers satisfaction rate from 1 to 10 and provide a very short straight-to-the-point area of improvement to the operator."},
]

def transcribe(audio):
    global messages
    
    segment_length = 60000
    # Open the audio file
    audio_file = AudioSegment.from_file(audio)
    # Get the duration of the audio file in milliseconds
    duration_ms = len(audio_file)

    # Calculate the number of segments needed
    num_segments = math.ceil(duration_ms / segment_length)

    # Create an empty string to hold the concatenated text
    all_text = ""
    # Split the audio file into segments

    for i in range(num_segments):
        start = i * segment_length
        end = min((i + 1) * segment_length, duration_ms)
        segment = audio_file[start:end]
        segment.export(f"segment_{i}.mp3", format="mp3")

    for i in range(num_segments):
        audio_file = open(f"segment_{i}.mp3", "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        all_text += transcript["text"]

    messages.append({"role": "user", "content": all_text})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    systems_message = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": systems_message})
    
    chat_transcript = ""
    for message in messages:
        if message['role'] != 'system':
            chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

    return systems_message

def progress_callback(portion, total):
    print(f"{portion}/{total}")

io = gr.Interface(fn=transcribe, inputs=gr.Audio(source="upload", type="filepath"), outputs="text", title="AI Auditor for Call Center's Quality Assurance", 
                  description="AI Alliance for Audio Analytics Team. Our project's objective is to conduct quality assurance on recorded calls, by transcribing the speech in the call to text using Whisper and then employing GPT-3 for sentiment analysis, summarisation, and feedback including areas for improvement. ",
                  examples=[["./Samples/SampleCall1.mp3"], ["./Samples/SampleCall2.mp3"], ["./Samples/SampleCall3.mp3"]],
                  interpretation="default")

io.launch(share=True)
