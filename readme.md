# Meeting Minutes Summariser

 An AI-powered tool that converts meeting audio into structured meeting minutes and a downloadable PDF.

 ## Features

 - Upload an MP3 meeting recording.
- Automatically transcribe the audio.
- Generate concise meeting minutes.
- Download the minutes as a PDF.

 ### Models

 - **Transcription:** `openai/whisper-medium.en`
- **Summarisation:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

 ## Requirements

 > **CUDA is required.** This project is designed to run with a CUDA-compatible NVIDIA GPU. CPU-only execution is not supported/recommended.

 - Python 3.10+
- [UV](<https://docs.astral.sh/uv/>)
- FFmpeg
- CUDA-compatible NVIDIA GPU
- Internet connection for downloading models

 ## Setup

```
git clone <repository-url>
cd <repository-directory>

uv sync
```

 Activate the virtual environment:

 ### Linux / macOS

```
source .venv/bin/activate
```

 ### Windows

```
.venv\Scripts\Activate.ps1
```

 ## Configuration

 Edit:

```
utils/config.py
```

 You can configure the models, input/output directories, and maximum tokens.


```
GRADIO_USER_NAME=set_desired_value
GRADIO_USER_PASS=set_desired_value
```
 This will be login credentials for Gradio UI

 ## Run

```
python app.py
```

 Or:

```
uv run python app.py
```

 ## Usage

 1. Upload an MP3 meeting recording.
2. The audio is transcribed using Whisper.
3. The transcript is summarised using DeepSeek.
4. A PDF containing the meeting minutes is generated.
5. Download the resulting PDF.

 **Input:** Meeting audio (`.mp3`)\
 **Output:** Meeting minutes (`.pdf`)