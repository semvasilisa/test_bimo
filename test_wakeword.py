import pyaudiowpatch as pyaudio
import numpy as np
from openwakeword.model import Model

# 1. Initialize OpenWakeWord with the built-in "hey_jarvis" model
owwModel = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

# 2. Audio Capture Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000          # 16kHz sampling rate required by openWakeWord
CHUNK = 1280           # 80ms chunk size required by openWakeWord

audio = pyaudio.PyAudio()
mic_stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

print("\n[BMO Engine] Wake word listener active! Say 'Hey Jarvis' to test...")

try:
    while True:
        # Read microphone stream
        raw_data = mic_stream.read(CHUNK, exception_on_overflow=False)
        
        # Convert binary audio buffer into numpy array
        audio_frame = np.frombuffer(raw_data, dtype=np.int16)

        # Feed frame to openWakeWord model
        prediction = owwModel.predict(audio_frame)

        # Check probability score for "hey_jarvis"
        score = owwModel.prediction_buffer["hey_jarvis"][-1]

        # Trigger if detection score passes 50% threshold
        if score > 0.5:
            print(f"\n⚡ [WAKE WORD DETECTED!] Score: {score:.2f} — BMO is listening...")
            owwModel.reset()

except KeyboardInterrupt:
    print("\nStopping wake word listener...")
    mic_stream.stop_stream()
    mic_stream.close()
    audio.terminate()