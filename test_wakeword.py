import pyaudiowpatch as pyaudio # handles audio input/output from microphone
import numpy as np
from openwakeword.model import Model # wake-word detection model
import os
import wave
import whisper

WAKE_WORD_MODEL = "./wakeword.onnx"
owwModel = Model(wakeword_models=[WAKE_WORD_MODEL])

# Transcribe Audio using Local Whisper Model
print("[BIMO Engine] Loading local Whisper model (tiny.en)...")
model = whisper.load_model("tiny.en")

# 2. Audio Capture Settings, here we configure microphone and tell python how to capture audio
AUDIO_FILE = "recorded_prompt.wav"
FORMAT = pyaudio.paInt16 # how each audio sample should be represented in memory (16-bit signed integer), these numbers describe the shape of the sound wave
CHANNELS = 1 # record one audio channel
RATE = 16000 # openWakeWord requiers audio sampled 16000 times per second
CHUNK = 1280 # it cuts microphone steam into 1280-sample frames (80ms chunks) for analysis
RECORD_SECONDS = 10

audio = pyaudio.PyAudio() # initialize the audio system
# open a stream to the microphone, this is where we will read audio input data from
mic_stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

try:
    while True:
        print("\n[BMO Engine] Wake word listener active! Say 'Hey BMO' to test...")

        # --- Read microphone stream - captures 80ms(1280 samples) of sound from mic, it's a raw binary audio data ---
        raw_data = mic_stream.read(CHUNK, exception_on_overflow=False)
            
        # --- Convert raw binary audio buffer into numpy array, so we go from [10101001 01011010 00101001 ...] to [-1234 5678 -2345 ...] ---
        audio_frame = np.frombuffer(raw_data, dtype=np.int16)

        # --- predict wake word ---
        prediction = owwModel.predict(audio_frame)
        for key,value in prediction.items():
            print(f"\r[BMO Engine] Wake word detection score: {value:.2f}", end="")

        # --- if detected -> record promt ---
        if prediction['wakeword'] > 0.8:
            print("\n[BMO Engine] Wake word detected! Recording your prompt...")
            recorded_audio = [] # here we will store the audio frames captured from the microphone

            # !!! add a delay befor start to record !!!

            # --- record audio for 10 seconds, each iteration(125) new microphone data ---
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                promt = mic_stream.read(CHUNK) # raw data, good for wave
                recorded_audio.append(promt)

            print("[BIMO Engine] Recording complete. Saving audio file...")

            # Save recorded audio buffer into a temporary .wav file
            wf = wave.open(AUDIO_FILE, 'wb')

            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)

            wf.writeframes(b''.join(recorded_audio)) # recorded audio is a list of byte strings, here we connect them into one byte stream
            wf.close()

            print("[BIMO Engine] Transcribing spoken audio...")
            # fp16=False suppresses the CPU warning and processes audio on CPU directly
            result = model.transcribe(AUDIO_FILE, fp16=False)

            print("\n--------------------------------------------------")
            print(f"⚡ [BIMO TRANSCRIBED TEXT]: {result['text'].strip()}")
            print("--------------------------------------------------\n")

except KeyboardInterrupt:
    mic_stream.stop_stream()
    mic_stream.close()
    audio.terminate()

    # Clean up local temp file
    if os.path.exists(AUDIO_FILE):
        os.remove(AUDIO_FILE)