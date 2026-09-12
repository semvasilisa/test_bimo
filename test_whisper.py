import os
import wave
import pyaudiowpatch as pyaudio
import whisper
import imageio_ffmpeg

# Explicitly add imageio-ffmpeg executable path to system PATH
ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

# 1. Configuration Settings
AUDIO_FILE = "recorded_prompt.wav"
RECORD_SECONDS = 5
SAMPLE_RATE = 16000
CHUNK = 1024

# 2. Record Audio from Microphone
print(f"\n[BIMO Engine] Speak into your mic now! Recording for {RECORD_SECONDS} seconds...")

audio = pyaudio.PyAudio()
mic_stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=CHUNK
)

frames = []
for _ in range(0, int(SAMPLE_RATE / CHUNK * RECORD_SECONDS)):
    data = mic_stream.read(CHUNK)
    frames.append(data)

print("[BIMO Engine] Recording complete. Saving audio file...")

mic_stream.stop_stream()
mic_stream.close()
audio.terminate()

# Save recorded audio buffer into a temporary .wav file
wf = wave.open(AUDIO_FILE, 'wb')
wf.setnchannels(1)
wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
wf.setframerate(SAMPLE_RATE)
wf.writeframes(b''.join(frames))
wf.close()

# 3. Transcribe Audio using Local Whisper Model
print("[BIMO Engine] Loading local Whisper model (tiny.en)...")
model = whisper.load_model("tiny.en")

print("[BIMO Engine] Transcribing spoken audio...")
# fp16=False suppresses the CPU warning and processes audio on CPU directly
result = model.transcribe(AUDIO_FILE, fp16=False)

print("\n--------------------------------------------------")
print(f"⚡ [BIMO TRANSCRIBED TEXT]: {result['text'].strip()}")
print("--------------------------------------------------\n")

# Clean up local temp file
if os.path.exists(AUDIO_FILE):
    os.remove(AUDIO_FILE)