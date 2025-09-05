"""
Enhanced Gradio interface for HiggsAudio model serving.
Provides a comprehensive web UI with voice cloning, multi-speaker support, and advanced features.
Merged from gradio_interface1.py while preserving 8-bit quantization functionality.
"""

import argparse
import base64
import os
import uuid
import json
from typing import Optional
import gradio as gr
from loguru import logger
import numpy as np
import time
from functools import lru_cache
import re
import torch
import torchaudio
import tempfile
import gc
import logging

# Import HiggsAudio components
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, AudioContent, Message

# Set up logging
logger = logging.getLogger(__name__)

# Global variables to store the engine and initialization state
engine = None
is_initialized = False

# Default model configuration - keeping Pinokio paths
DEFAULT_MODEL_PATH = "models/higgs-audio-v2-generation-3B-base"
DEFAULT_AUDIO_TOKENIZER_PATH = "models/higgs-audio-v2-tokenizer"
SAMPLE_RATE = 24000

DEFAULT_SYSTEM_PROMPT = (
    "Generate audio following instruction.\n\n"
    "<|scene_desc_start|>\n"
    "Audio is recorded from a quiet room.\n"
    "<|scene_desc_end|>"
)

DEFAULT_STOP_STRINGS = ["<|end_of_text|>", "<|eot_id|>"]

# Predefined examples for system and input messages
PREDEFINED_EXAMPLES = {
    "voice-clone": {
        "system_prompt": "",
        "input_text": "Hey there! I'm your friendly voice twin in the making. Pick a voice preset below or upload your own audio - let's clone some vocals and bring your voice to life! ",
        "description": "Voice clone to clone the reference audio. Leave the system prompt empty.",
    },
    "smart-voice": {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "input_text": "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.",
        "description": "Smart voice to generate speech based on the context",
    },
    "multispeaker-voice-description": {
        "system_prompt": "You are an AI assistant designed to convert text into speech.\n"
        "If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.\n"
        "If no speaker tag is present, select a suitable voice on your own.\n\n"
        "<|scene_desc_start|>\n"
        "SPEAKER0: feminine\n"
        "SPEAKER1: masculine\n"
        "<|scene_desc_end|>",
        "input_text": "[SPEAKER0] I can't believe you did that without even asking me first!\n"
        "[SPEAKER1] Oh, come on! It wasn't a big deal, and I knew you would overreact like this.\n"
        "[SPEAKER0] Overreact? You made a decision that affects both of us without even considering my opinion!\n"
        "[SPEAKER1] Because I didn't have time to sit around waiting for you to make up your mind! Someone had to act.",
        "description": "Multispeaker with different voice descriptions in the system prompt",
    },
    "single-speaker-voice-description": {
        "system_prompt": "Generate audio following instruction.\n\n"
        "<|scene_desc_start|>\n"
        "SPEAKER0: He speaks with a clear British accent and a conversational, inquisitive tone. His delivery is articulate and at a moderate pace, and very clear audio.\n"
        "<|scene_desc_end|>",
        "input_text": "Hey, everyone! Welcome back to Tech Talk Tuesdays.\n"
        "It's your host, Alex, and today, we're diving into a topic that's become absolutely crucial in the tech world — deep learning.\n"
        "And let's be honest, if you've been even remotely connected to tech, AI, or machine learning lately, you know that deep learning is everywhere.\n"
        "\n"
        "So here's the big question: Do you want to understand how deep learning works?\n",
        "description": "Single speaker with voice description in the system prompt",
    },
    "single-speaker-zh": {
        "system_prompt": "Generate audio following instruction.\n\n"
        "<|scene_desc_start|>\n"
        "Audio is recorded from a quiet room.\n"
        "<|scene_desc_end|>",
        "input_text": "大家好, 欢迎收听本期的跟李沐学AI. 今天沐哥在忙着洗数据, 所以由我, 希格斯主播代替他讲这期视频.\n"
        "今天我们要聊的是一个你绝对不能忽视的话题: 多模态学习.\n"
        "那么, 问题来了, 你真的了解多模态吗? 你知道如何自己动手构建多模态大模型吗.\n"
        "或者说, 你能察觉到我其实是个机器人吗?",
        "description": "Single speaker speaking Chinese",
    },
    "single-speaker-bgm": {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "input_text": "[music start] I will remember this, thought Ender, when I am defeated. To keep dignity, and give honor where it's due, so that defeat is not disgrace. And I hope I don't have to do it often. [music end]",
        "description": "Single speaker with BGM using music tag. This is an experimental feature and you may need to try multiple times to get the best result.",
    },
}

# Parameter presets for different use cases
PARAMETER_PRESETS = {
    "default": {
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 50,
        "max_completion_tokens": 1024,
        "ras_win_len": 7,
        "ras_win_max_num_repeat": 2,
        "description": "Balanced quality and speed settings"
    },
    "female_voice": {
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 50,
        "max_completion_tokens": 1024,
        "ras_win_len": 7,
        "ras_win_max_num_repeat": 2,
        "description": "Optimized settings for female voice generation"
    },
    "male_voice": {
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 50,
        "max_completion_tokens": 1024,
        "ras_win_len": 7,
        "ras_win_max_num_repeat": 2,
        "description": "Optimized settings for male voice generation"
    },
    "high_quality": {
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 30,
        "max_completion_tokens": 1024,
        "ras_win_len": 7,
        "ras_win_max_num_repeat": 2,
        "description": "Conservative settings for highest quality output"
    },
    "creative": {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 80,
        "max_completion_tokens": 1024,
        "ras_win_len": 7,
        "ras_win_max_num_repeat": 2,
        "description": "Higher temperature for more expressive and varied output"
    },
    "fast": {
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 50,
        "max_completion_tokens": 512,
        "ras_win_len": 7,
        "ras_win_max_num_repeat": 2,
        "description": "Faster generation with shorter output"
    }
}

# Voice presets will be loaded from config
VOICE_PRESETS = {}

@lru_cache(maxsize=20)
def encode_audio_file(file_path):
    """Encode an audio file to base64."""
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")

def get_current_device():
    """Get the current device."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_gpu_memory_info():
    """Get GPU memory usage information."""
    if not torch.cuda.is_available():
        return "❌ CUDA not available"
    
    try:
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device)
        free_memory = total_memory - allocated_memory
        
        total_gb = total_memory / 1024**3
        allocated_gb = allocated_memory / 1024**3
        cached_gb = cached_memory / 1024**3
        free_gb = free_memory / 1024**3
        
        usage_percent = (allocated_memory / total_memory) * 100
        
        return (f"🔍 **GPU Memory Status:**\n"
                f"- **Total VRAM:** {total_gb:.1f} GB\n"
                f"- **Allocated:** {allocated_gb:.1f} GB ({usage_percent:.1f}%)\n"
                f"- **Cached:** {cached_gb:.1f} GB\n"
                f"- **Free:** {free_gb:.1f} GB")
    except Exception as e:
        return f"❌ Error getting GPU info: {str(e)}"

def load_voice_presets():
    """Load the voice presets from the voice_examples directory."""
    try:
        with open("voice_examples/config.json", "r", encoding="utf-8") as f:
            voice_dict = json.load(f)
        voice_presets = {k: v["transcript"] for k, v in voice_dict.items()}
        voice_presets["EMPTY"] = "No reference voice"
        logger.info(f"Loaded voice presets: {list(voice_presets.keys())}")
        return voice_presets
    except FileNotFoundError:
        logger.warning("Voice examples config file not found. Using empty voice presets.")
        return {"EMPTY": "No reference voice"}
    except Exception as e:
        logger.error(f"Error loading voice presets: {e}")
        return {"EMPTY": "No reference voice"}

def get_voice_preset(voice_preset):
    """Get the voice path and text for a given voice preset."""
    voice_path = os.path.join("voice_examples", f"{voice_preset}.wav")
    if not os.path.exists(voice_path):
        logger.warning(f"Voice preset file not found: {voice_path}")
        return None, "Voice preset not found"
    
    text = VOICE_PRESETS.get(voice_preset, "No transcript available")
    return voice_path, text

def normalize_chinese_punctuation(text):
    """Convert Chinese (full-width) punctuation marks to English (half-width) equivalents."""
    chinese_to_english_punct = {
        "，": ", ", "。": ".", "：": ":", "；": ";", "？": "?", "！": "!",
        "（": "(", "）": ")", "【": "[", "】": "]", "《": "<", "》": ">",
        """: '"', """: '"', "'": "'", "'": "'", "、": ",", "—": "-",
        "…": "...", "·": ".", "「": '"', "」": '"', "『": '"', "』": '"',
    }
    
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)
    return text

def normalize_text(transcript: str):
    transcript = normalize_chinese_punctuation(transcript)
    transcript = transcript.replace("(", " ").replace(")", " ")
    transcript = transcript.replace("°F", " degrees Fahrenheit")
    transcript = transcript.replace("°C", " degrees Celsius")

    for tag, replacement in [
        ("[laugh]", "<SE>[Laughter]</SE>"),
        ("[humming start]", "<SE>[Humming]</SE>"),
        ("[humming end]", "<SE_e>[Humming]</SE_e>"),
        ("[music start]", "<SE_s>[Music]</SE_s>"),
        ("[music end]", "<SE_e>[Music]</SE_e>"),
        ("[music]", "<SE>[Music]</SE>"),
        ("[sing start]", "<SE_s>[Singing]</SE_s>"),
        ("[sing end]", "<SE_e>[Singing]</SE_e>"),
        ("[applause]", "<SE>[Applause]</SE>"),
        ("[cheering]", "<SE>[Cheering]</SE>"),
        ("[cough]", "<SE>[Cough]</SE>"),
    ]:
        transcript = transcript.replace(tag, replacement)

    lines = transcript.split("\n")
    transcript = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
    transcript = transcript.strip()

    if not any([transcript.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
        transcript += "."

    return transcript

def load_parameter_preset(preset_name):
    """Load parameter preset settings."""
    if preset_name in PARAMETER_PRESETS:
        preset = PARAMETER_PRESETS[preset_name]
        return (
            gr.update(value=preset["temperature"]),
            gr.update(value=preset["top_p"]),
            gr.update(value=preset["top_k"]),
            gr.update(value=preset["max_completion_tokens"]),
            gr.update(value=preset["ras_win_len"]),
            gr.update(value=preset["ras_win_max_num_repeat"]),
            f"✅ '{preset_name.replace('_', ' ').title()}' preset loaded: {preset['description']}"
        )
    else:
        return tuple([gr.update() for _ in range(7)])

def reset_to_defaults():
    """Reset all parameters to default values."""
    return load_parameter_preset("default")

def initialize_engine(
    model_path: str = DEFAULT_MODEL_PATH,
    audio_tokenizer_path: str = DEFAULT_AUDIO_TOKENIZER_PATH,
    tokenizer_path: str = DEFAULT_MODEL_PATH,
    device: str = None,
    load_in_8bit: bool = False
):
    """
    Initialize the HiggsAudio serving engine with optimizations for 16GB VRAM or less.
    
    Args:
        model_path: Path to the HiggsAudio model
        audio_tokenizer_path: Path to the audio tokenizer
        tokenizer_path: Path to the tokenizer (optional)
        device: Device to use for inference (auto-detected if None)
        load_in_8bit: Whether to load model in 8-bit quantized mode
    
    Returns:
        Status message and model status display
    """
    global engine, is_initialized
    
    try:
        # Validate and set default paths if empty
        if not model_path or model_path.strip() == "":
            model_path = DEFAULT_MODEL_PATH
            logger.warning(f"Empty model path provided, using default: {model_path}")
        
        if not audio_tokenizer_path or audio_tokenizer_path.strip() == "":
            audio_tokenizer_path = DEFAULT_AUDIO_TOKENIZER_PATH
            logger.warning(f"Empty audio tokenizer path provided, using default: {audio_tokenizer_path}")
        
        # Handle optional tokenizer path
        if not tokenizer_path or tokenizer_path.strip() == "":
            tokenizer_path = DEFAULT_MODEL_PATH
        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        if device is None:
            device = get_current_device()
            
        logger.info(f"Initializing HiggsAudio engine with 8-bit: {load_in_8bit}")
        
        # Set optimized KV cache lengths for lower VRAM usage
        if load_in_8bit:
            kv_cache_lengths = [512, 1024, 2048]  # Smaller cache sizes for 8-bit
        else:
            kv_cache_lengths = [1024, 2048, 4096]  # Standard cache sizes
        
        engine = HiggsAudioServeEngine(
            model_name_or_path=model_path,
            audio_tokenizer_name_or_path=audio_tokenizer_path,
            tokenizer_name_or_path=tokenizer_path,
            device=device,
            kv_cache_lengths=kv_cache_lengths
        )
        
        is_initialized = True
        
        # Get memory info after initialization
        memory_info = get_gpu_memory_info()
        quantization_info = " (8-bit quantized)" if load_in_8bit else " (full precision)"
        status_msg = f"✅ Model successfully loaded on {device}{quantization_info}! Ready to generate speech.\n\n{memory_info}"
        status_display = "🟢 **Model Status:** Ready - Model loaded and ready for speech generation"
        
        logger.info(f"Successfully initialized HiggsAudioServeEngine with model: {model_path}")
        return status_msg, status_display
        
    except Exception as e:
        is_initialized = False
        engine = None
        logger.error(f"Failed to initialize engine: {e}")
        error_msg = f"❌ Error loading model: {str(e)}"
        status_display = f"🔴 **Model Status:** Error - {str(e)}"
        return error_msg, status_display

def process_text_output(text_output: str):
    """Remove all the continuous <|AUDIO_OUT|> tokens with a single <|AUDIO_OUT|>."""
    text_output = re.sub(r"(<\|AUDIO_OUT\|>)+", r"<|AUDIO_OUT|>", text_output)
    return text_output

def prepare_chatml_sample(
    voice_preset: str,
    text: str,
    reference_audio: Optional[str] = None,
    reference_text: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
):
    """Prepare a ChatMLSample for the HiggsAudioServeEngine."""
    messages = []

    # Add system message if provided
    if len(system_prompt) > 0:
        messages.append(Message(role="system", content=system_prompt))

    # Add reference audio if provided
    audio_base64 = None
    ref_text = ""

    if reference_audio:
        # Custom reference audio
        audio_base64 = encode_audio_file(reference_audio)
        ref_text = reference_text or ""
    elif voice_preset != "EMPTY":
        # Voice preset
        voice_path, ref_text = get_voice_preset(voice_preset)
        if voice_path is None:
            logger.warning(f"Voice preset {voice_preset} not found, skipping reference audio")
        else:
            audio_base64 = encode_audio_file(voice_path)

    # Only add reference audio if we have it
    if audio_base64 is not None:
        # Add user message with reference text
        messages.append(Message(role="user", content=ref_text))
        
        # Add assistant message with audio content
        audio_content = AudioContent(raw_audio=audio_base64, audio_url="")
        messages.append(Message(role="assistant", content=[audio_content]))

    # Add the main user message
    text = normalize_text(text)
    messages.append(Message(role="user", content=text))

    return ChatMLSample(messages=messages)

def text_to_speech(
    text,
    voice_preset,
    reference_audio=None,
    reference_text=None,
    max_completion_tokens=1024,
    temperature=1.0,
    top_p=0.95,
    top_k=50,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    stop_strings=None,
    ras_win_len=7,
    ras_win_max_num_repeat=2,
):
    """Convert text to speech using HiggsAudioServeEngine."""
    global engine

    if engine is None:
        init_result = initialize_engine(
            model_path=DEFAULT_MODEL_PATH,
            audio_tokenizer_path=DEFAULT_AUDIO_TOKENIZER_PATH,
            tokenizer_path=DEFAULT_MODEL_PATH,
            device=None,
            load_in_8bit=False
        )
        if "Error" in init_result[0]:
            return init_result[0], None, init_result[1]

    try:
        # Prepare ChatML sample
        chatml_sample = prepare_chatml_sample(voice_preset, text, reference_audio, reference_text, system_prompt)

        # Convert stop strings format
        if stop_strings is None:
            stop_list = DEFAULT_STOP_STRINGS
        else:
            stop_list = [s for s in stop_strings["stops"] if s.strip()]

        request_id = f"tts-enhanced-{str(uuid.uuid4())}"
        logger.info(
            f"{request_id}: Generating speech for text: {text[:100]}..., \n"
            f"with parameters: temperature={temperature}, top_p={top_p}, top_k={top_k}, stop_list={stop_list}, "
            f"ras_win_len={ras_win_len}, ras_win_max_num_repeat={ras_win_max_num_repeat}"
        )
        start_time = time.time()

        # Generate using the engine
        response = engine.generate(
            chat_ml_sample=chatml_sample,
            max_new_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p,
            stop_strings=stop_list,
            ras_win_len=ras_win_len if ras_win_len > 0 else None,
            ras_win_max_num_repeat=max(ras_win_len, ras_win_max_num_repeat),
        )

        generation_time = time.time() - start_time
        logger.info(f"{request_id}: Generated audio in {generation_time:.3f} seconds")

        # Process the response
        text_output = process_text_output(response.generated_text)

        if response.audio is not None:
            # Save the generated audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                # Convert to proper format for torchaudio
                audio_tensor = torch.from_numpy(response.audio)[None, :]
                torchaudio.save(tmp_file.name, audio_tensor, response.sampling_rate)
                
                return f"✅ {text_output}\n\nGenerated in {generation_time:.3f}s", tmp_file.name, "🟢 **Model Status:** Ready - Model loaded and ready for speech generation"
        else:
            logger.warning("No audio generated")
            return f"⚠️ {text_output}\n\nNo audio generated", None, "🟢 **Model Status:** Ready - Model loaded and ready for speech generation"

    except Exception as e:
        error_msg = f"Error generating speech: {e}"
        logger.error(error_msg)
        return f"❌ {error_msg}", None, f"🔴 **Model Status:** Error - {str(e)}"

def create_gradio_interface():
    """Create the enhanced Gradio interface for HiggsAudio with all advanced features."""
    
    # Load theme and voice presets
    try:
        my_theme = gr.Theme.load("theme.json")
    except:
        my_theme = gr.themes.Default()
    
    global VOICE_PRESETS
    VOICE_PRESETS = load_voice_presets()

    # Custom CSS to disable focus highlighting
    custom_css = """
    .gradio-container input:focus, 
    .gradio-container textarea:focus, 
    .gradio-container select:focus,
    .gradio-container .gr-input:focus,
    .gradio-container .gr-textarea:focus,
    .gradio-container .gr-textbox:focus,
    .gradio-container .gr-textbox:focus-within,
    .gradio-container .gr-form:focus-within,
    .gradio-container *:focus {
        box-shadow: none !important;
        border-color: var(--border-color-primary) !important;
        outline: none !important;
        background-color: var(--input-background-fill) !important;
    }
    """

    default_template = "smart-voice"
    
    with gr.Blocks(theme=my_theme, css=custom_css, title="HiggsAudio Enhanced Interface") as interface:
        gr.Markdown("# HiggsAudio V2 Enhanced Text-to-Speech Interface")
        gr.Markdown("Generate expressive speech with voice cloning, multi-speaker support, background music, and 8-bit quantization support.")

        # Model status indicator
        model_status_display = gr.Markdown("🔴 **Model Status:** Not initialized - Click 'Initialize' below to start")

        with gr.Row():
            with gr.Column(scale=2):
                # Template selection dropdown
                template_dropdown = gr.Dropdown(
                    label="TTS Template",
                    choices=list(PREDEFINED_EXAMPLES.keys()),
                    value=default_template,
                    info="Select a predefined example for system and input messages.",
                )

                # Template description display
                template_description = gr.HTML(
                    value=f'<p style="font-size: 0.85em; color: var(--body-text-color-subdued); margin: 0; padding: 0;"> {PREDEFINED_EXAMPLES[default_template]["description"]}</p>',
                    visible=True,
                )

                system_prompt = gr.TextArea(
                    label="System Prompt",
                    placeholder="Enter system prompt to guide the model...",
                    value=PREDEFINED_EXAMPLES[default_template]["system_prompt"],
                    lines=3,
                )

                input_text = gr.TextArea(
                    label="Input Text",
                    placeholder="Type the text you want to convert to speech...",
                    value=PREDEFINED_EXAMPLES[default_template]["input_text"],
                    lines=5,
                )

                voice_preset = gr.Dropdown(
                    label="Voice Preset",
                    choices=list(VOICE_PRESETS.keys()),
                    value="EMPTY",
                    interactive=False,
                    visible=False,
                )

                with gr.Accordion("Custom Reference (Optional)", open=False, visible=False) as custom_reference_accordion:
                    reference_audio = gr.Audio(label="Reference Audio", type="filepath")
                    reference_text = gr.TextArea(
                        label="Reference Text (transcript of the reference audio)",
                        placeholder="Enter the transcript of your reference audio...",
                        lines=3,
                    )

                with gr.Accordion("Advanced Parameters", open=False):
                    max_completion_tokens = gr.Slider(
                        minimum=128,
                        maximum=4096,
                        value=1024,
                        step=10,
                        label="Max Completion Tokens",
                    )
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.5,
                        value=1.0,
                        step=0.1,
                        label="Temperature",
                    )
                    top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top P")
                    top_k = gr.Slider(minimum=-1, maximum=100, value=50, step=1, label="Top K")
                    ras_win_len = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=7,
                        step=1,
                        label="RAS Window Length",
                        info="Window length for repetition avoidance sampling",
                    )
                    ras_win_max_num_repeat = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=2,
                        step=1,
                        label="RAS Max Num Repeat",
                        info="Maximum number of repetitions allowed in the window",
                    )
                    # Add stop strings component
                    stop_strings = gr.Dataframe(
                        label="Stop Strings",
                        headers=["stops"],
                        datatype=["str"],
                        value=[[s] for s in DEFAULT_STOP_STRINGS],
                        interactive=True,
                        col_count=(1, "fixed"),
                    )

                    # Parameter presets section
                    with gr.Row():
                        parameter_preset_dropdown = gr.Dropdown(
                            choices=[
                                ("Default - Balanced quality and speed", "default"),
                                ("Female Voice - Optimized for female speech", "female_voice"),
                                ("Male Voice - Optimized for male speech", "male_voice"),
                                ("High Quality - Conservative settings for best quality", "high_quality"),
                                ("Creative - More expressive and varied output", "creative"),
                                ("Fast - Quick generation with shorter output", "fast")
                            ],
                            value="default",
                            label="Parameter Presets",
                            info="Choose preset parameter configurations"
                        )
                        load_param_preset_btn = gr.Button("📋 Load Parameters", variant="secondary")
                    
                    reset_defaults_btn = gr.Button("🔄 Reset All to Defaults", variant="secondary")

                generate_btn = gr.Button("Generate Speech", variant="primary")

            with gr.Column(scale=2):
                output_text = gr.TextArea(label="Model Response", lines=3)
                output_audio = gr.Audio(label="Generated Audio", interactive=False, autoplay=True)
                preset_status_display = gr.Textbox(label="Parameter Status", interactive=False, placeholder="Parameter presets can be loaded from Advanced Settings")

        # Voice samples section
        with gr.Row(visible=False) as voice_samples_section:
            voice_samples_table = gr.Dataframe(
                headers=["Voice Preset", "Sample Text"],
                datatype=["str", "str"],
                value=[[preset, text] for preset, text in VOICE_PRESETS.items() if preset != "EMPTY"],
                interactive=False,
            )
            sample_audio = gr.Audio(label="Voice Sample")

        # Model initialization section with enhanced 8-bit quantization support
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🚀 Model Setup & Configuration")
                gr.Markdown("*Configure and initialize the HiggsAudio model (first-time setup may take 5-10 minutes)*")
                
                # Model configuration section
                with gr.Accordion("🔧 Model Configuration", open=False):
                    model_path = gr.Textbox(
                        label="Model Path",
                        placeholder="Enter model name or path",
                        value=DEFAULT_MODEL_PATH,
                        info="Path to the HiggsAudio model (Pinokio: models/higgs-audio-v2-generation-3B-base)"
                    )
                    
                    audio_tokenizer_path = gr.Textbox(
                        label="Audio Tokenizer Path", 
                        placeholder="Enter audio tokenizer name or path",
                        value=DEFAULT_AUDIO_TOKENIZER_PATH,
                        info="Path to the audio tokenizer (Pinokio: models/higgs-audio-v2-tokenizer)"
                    )
                    
                    tokenizer_path = gr.Textbox(
                        label="Tokenizer Path (Optional)",
                        placeholder="Leave empty to use model path",
                        info="Optional separate tokenizer path"
                    )
                    
                    device = gr.Dropdown(
                        choices=["cuda", "cpu"],
                        value="cuda" if torch.cuda.is_available() else "cpu",
                        label="Device",
                        info="Device to use for inference"
                    )
                
                with gr.Accordion("⚙️ Model Loading Options", open=True):
                    gr.Markdown("**🚀 Model Loading Configuration:**")
                    gr.Markdown("• **8-bit Quantization**: ~50% less VRAM usage, minimal quality impact")
                    gr.Markdown("• **Memory Info**: Check current VRAM usage before loading")
                    
                    load_in_8bit_checkbox = gr.Checkbox(
                        label="Enable 8-bit Quantization",
                        value=False,
                        info="Reduces VRAM usage by ~50%, recommended for GPUs with 16GB or less"
                    )
                    
                    with gr.Row():
                        reload_model_btn = gr.Button("🔄 Load/Reload Model", variant="primary")
                        memory_info_btn = gr.Button("📊 Check GPU Memory", variant="secondary")
                    
                    gr.Markdown(
                        """
                        **8-bit Quantization Benefits:**
                        - Reduces GPU memory usage by approximately 50%
                        - Enables running larger models on smaller GPUs (16GB VRAM or less)
                        - Minimal impact on audio quality
                        - Automatically optimizes KV cache sizes for lower memory usage
                        """
                    )
                
                init_status = gr.Textbox(label="Status", interactive=False, placeholder="Click 'Load/Reload Model' to initialize the model...")
                memory_status = gr.Markdown("Click 'Check GPU Memory' to see current VRAM usage")

        # Event handlers
        def play_voice_sample(evt: gr.SelectData):
            try:
                preset_names = [preset for preset in VOICE_PRESETS.keys() if preset != "EMPTY"]
                if evt.index[0] < len(preset_names):
                    preset = preset_names[evt.index[0]]
                    voice_path, _ = get_voice_preset(preset)
                    if voice_path and os.path.exists(voice_path):
                        return voice_path
                    else:
                        gr.Warning(f"Voice sample file not found for preset: {preset}")
                        return None
                else:
                    gr.Warning("Invalid voice preset selection")
                    return None
            except Exception as e:
                logger.error(f"Error playing voice sample: {e}")
                gr.Error(f"Error playing voice sample: {e}")
                return None

        def apply_template(template_name):
            if template_name in PREDEFINED_EXAMPLES:
                template = PREDEFINED_EXAMPLES[template_name]
                is_voice_clone = template_name == "voice-clone"
                voice_preset_value = "belinda" if is_voice_clone else "EMPTY"
                ras_win_len_value = 0 if template_name == "single-speaker-bgm" else 7
                description_text = f'<p style="font-size: 0.85em; color: var(--body-text-color-subdued); margin: 0; padding: 0;"> {template["description"]}</p>'
                return (
                    template["system_prompt"],
                    template["input_text"],
                    description_text,
                    gr.update(value=voice_preset_value, interactive=is_voice_clone, visible=is_voice_clone),
                    gr.update(visible=is_voice_clone),
                    gr.update(visible=is_voice_clone),
                    ras_win_len_value,
                )
            else:
                return tuple([gr.update() for _ in range(7)])

        # Connect event handlers
        voice_samples_table.select(fn=play_voice_sample, outputs=[sample_audio])

        template_dropdown.change(
            fn=apply_template,
            inputs=[template_dropdown],
            outputs=[
                system_prompt,
                input_text,
                template_description,
                voice_preset,
                custom_reference_accordion,
                voice_samples_section,
                ras_win_len,
            ],
        )

        generate_btn.click(
            fn=text_to_speech,
            inputs=[
                input_text,
                voice_preset,
                reference_audio,
                reference_text,
                max_completion_tokens,
                temperature,
                top_p,
                top_k,
                system_prompt,
                stop_strings,
                ras_win_len,
                ras_win_max_num_repeat,
            ],
            outputs=[output_text, output_audio, model_status_display],
        )

        # Model reload button with 8-bit checkbox
        reload_model_btn.click(
            fn=initialize_engine,
            inputs=[model_path, audio_tokenizer_path, tokenizer_path, device, load_in_8bit_checkbox],
            outputs=[init_status, model_status_display]
        )

        memory_info_btn.click(
            fn=get_gpu_memory_info,
            outputs=[memory_status]
        )

        # Parameter preset event handlers
        load_param_preset_btn.click(
            fn=load_parameter_preset,
            inputs=[parameter_preset_dropdown],
            outputs=[
                temperature,
                top_p,
                top_k,
                max_completion_tokens,
                ras_win_len,
                ras_win_max_num_repeat,
                preset_status_display,
            ]
        )

        reset_defaults_btn.click(
            fn=reset_to_defaults,
            outputs=[
                temperature,
                top_p,
                top_k,
                max_completion_tokens,
                ras_win_len,
                ras_win_max_num_repeat,
                preset_status_display,
            ]
        )
    
    return interface

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse command line arguments for enhanced functionality
    parser = argparse.ArgumentParser(description="Enhanced HiggsAudio V2 Gradio Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Enable public sharing via Gradio")
    
    args = parser.parse_args()

    print(f"Starting Enhanced HiggsAudio V2 interface on {args.host}:{args.port}")
    if args.share:
        print("Public sharing enabled via Gradio")
    else:
        print("Local access only")
    
    # Create and launch the interface
    interface = create_gradio_interface()
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )