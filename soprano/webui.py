#!/usr/bin/env python3
"""
Gradio Web Interface for Soprano TTS
"""

import gradio as gr
import torch
import sys
import os
import asyncio
import logging
import numpy as np
import socket
import time
import threading
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soprano import SopranoTTS

logging.getLogger('asyncio').setLevel(logging.CRITICAL)

def custom_exception_handler(loop, context):
    exception = context.get('exception')

    if exception and not (isinstance(exception, ConnectionResetError) and "forcibly closed" in str(exception)):
        print(f"AsyncIO Exception: {context.get('message')}")
        exc = context.get('exception')
        if exc:
            traceback.print_exception(type(exc), exc, exc.__traceback__)

    if exception and isinstance(exception, ConnectionResetError) and "forcibly closed" in str(exception):
        pass
    else:
        loop.default_exception_handler(context)

if sys.platform.startswith("win"):
    try:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(custom_exception_handler)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        loop.set_exception_handler(custom_exception_handler)
        asyncio.set_event_loop(loop)

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("PyAudio not found. Install it with 'pip install pyaudio' for real-time audio streaming.")

current_stream = None
current_pyaudio_instance = None
stream_lock = threading.Lock()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 32000

print("Loading Soprano TTS model...")
model = SopranoTTS(
    backend="auto",
    device=DEVICE,
    cache_size_mb=100,
    decoder_batch_size=1,
)
print("Model loaded successfully!")


async def generate_speech(
    text: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> tuple:
    if not text.strip():
        return None, "Please enter some text to generate speech."

    try:
        start_time = time.perf_counter()

        audio = model.infer(
            text,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        gen_time = time.perf_counter() - start_time

        audio_np = audio.cpu().numpy()

        if audio_np.size == 0:
            return None, "✗ Error: Generated audio is empty."

        max_val = np.max(np.abs(audio_np))
        if max_val > 1.0:
            audio_np = audio_np / max_val

        audio_int16 = (audio_np * 32767).astype(np.int16)

        audio_seconds = len(audio_np) / SAMPLE_RATE
        rtf = audio_seconds / gen_time if gen_time > 0 else float("inf")

        status = (
            f"✓ Generated {audio_seconds:.2f} s audio | "
            f"Generation time: {gen_time:.3f} s "
            f"({rtf:.2f}x realtime)"
        )

        return (SAMPLE_RATE, audio_int16), status

    except ConnectionResetError:
        return None, "✗ Connection error during generation. Please try again."
    except Exception as e:
        return None, f"✗ Error: {str(e)}"


class AudioStreamer:

    def __init__(self):
        self.stream = None
        self.pyaudio_instance = None
        self.is_playing = False

    def cleanup(self):
        if self.stream:
            try:
                if hasattr(self.stream, 'is_active') and self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except Exception:
                pass
            self.pyaudio_instance = None

        self.is_playing = False

    def play_audio_chunk(self, audio_chunk):
        if self.stream and self.is_playing:
            audio_np = audio_chunk.cpu().numpy()

            if len(audio_np) > 1:
                if len(audio_np) == 2:
                    smoothed = np.array([audio_np[0], audio_np[1]])
                elif len(audio_np) == 3:
                    smoothed = np.array([
                        audio_np[0],
                        (audio_np[0] + audio_np[1] + audio_np[2]) / 3,
                        audio_np[2]
                    ])
                else:
                    smoothed = np.zeros_like(audio_np)
                    smoothed[0] = audio_np[0]
                    smoothed[-1] = audio_np[-1]
                    if len(audio_np) > 2:
                        smoothed[1:-1] = (audio_np[:-2] + audio_np[1:-1] + audio_np[2:]) / 3
                audio_np = smoothed

            audio_np = np.clip(audio_np, -1.0, 1.0)

            audio_int16 = (np.tanh(audio_np) * 32767).astype(np.int16)

            self.stream.write(audio_int16.tobytes())
            return len(audio_int16)

        return 0


async def speak_realtime(
    text: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
    global current_stream, current_pyaudio_instance

    if not text.strip():
        return "Please enter some text to speak."

    if not PYAUDIO_AVAILABLE:
        return "PyAudio is not available. Install it with 'pip install pyaudio' for real-time audio streaming."

    with stream_lock:
        audio_streamer = AudioStreamer()

        current_stream = None
        current_pyaudio_instance = None

        try:
            audio_streamer.pyaudio_instance = pyaudio.PyAudio()

            audio_streamer.stream = audio_streamer.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                output=True,
                frames_per_buffer=2048  # Smaller buffer for more responsive streaming
            )
            audio_streamer.is_playing = True

            current_stream = audio_streamer.stream
            current_pyaudio_instance = audio_streamer.pyaudio_instance

            start_time = time.perf_counter()

            # Real-time streaming: immediately feed text into synthesis engine
            # Begin generating audio in small chunks without waiting for full text processing
            stream_gen = model.infer_stream(
                text,
                chunk_size=1,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            total_samples = 0

            # As each chunk of audio is produced, stream and play it back immediately
            # while the rest of the text is still being converted
            for audio_chunk in stream_gen:
                # Check if stream is still active to maintain continuous, live speech output
                if (not audio_streamer.is_playing or
                    not (hasattr(audio_streamer.stream, 'is_active') and audio_streamer.stream.is_active())):
                    break

                # Stream and play audio chunks as they become available
                samples_written = audio_streamer.play_audio_chunk(audio_chunk)
                total_samples += samples_written

            audio_streamer.cleanup()

            current_stream = None
            current_pyaudio_instance = None

            gen_time = time.perf_counter() - start_time
            audio_seconds = total_samples / SAMPLE_RATE
            rtf = audio_seconds / gen_time if gen_time > 0 else float("inf")

            status = (
                f"✓ Finished speaking {audio_seconds:.2f} s audio | "
                f"Playback time: {gen_time:.3f} s "
                f"({rtf:.2f}x realtime)"
            )

            return status

        except ConnectionResetError:
            audio_streamer.cleanup()

            current_stream = None
            current_pyaudio_instance = None

            return "✗ Connection error during real-time playback. Please try again."
        except Exception as e:
            audio_streamer.cleanup()

            current_stream = None
            current_pyaudio_instance = None

            return f"✗ Error during real-time playback: {str(e)}"


def create_gradio_interface():
    with gr.Blocks(title="Soprano TTS") as demo:
        active_function = gr.State(value="ready")

        gr.Markdown(
            f"""
    # 🎵 Soprano TTS

    **Running on: {DEVICE.upper()}**

    Soprano is an ultra-lightweight, open-source text-to-speech (TTS) model designed for real-time,
    high-fidelity speech synthesis at unprecedented speed. Soprano can achieve **<15 ms streaming latency**
    and up to **2000x real-time generation**, all while being easy to deploy at **<1 GB VRAM usage**.

    <br>

    <div style="display: flex; justify-content: center; gap: 20px; margin: 15px 0;">
        <a href="https://github.com/ekwek1/soprano" target="_blank" style="text-decoration: none; color: inherit;">Git Hub</a>
        <a href="https://huggingface.co/spaces/ekwek/Soprano-TTS" target="_blank" style="text-decoration: none; color: inherit;">Model Demo</a>
        <a href="https://huggingface.co/ekwek/Soprano-80M" target="_blank" style="text-decoration: none; color: inherit;"> Model Weights </a>
    </div>
    """
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Enter text here...",
                    value="Soprano is an extremely lightweight text to speech model designed to produce highly realistic speech at unprecedented speed.",
                    lines=5,
                    max_lines=10,
                )

                with gr.Accordion("Advanced Settings", open=False):
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.3,
                        step=0.05,
                        label="Temperature",
                    )

                    top_p = gr.Slider(
                        minimum=0.5,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                        label="Top P",
                    )

                    repetition_penalty = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.2,
                        step=0.1,
                        label="Repetition Penalty",
                    )

                with gr.Row():
                    generate_btn = gr.Button("Generate Speech", variant="primary", size="lg")
                    speak_btn = gr.Button("Speak", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear", variant="secondary", size="lg")

            with gr.Column(scale=1):
                audio_output = gr.Audio(
                    label="Generated Speech",
                    type="numpy",
                    autoplay=True,
                    streaming=True
                )

                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3,
                    max_lines=10
                )

        gr.Examples(
            examples=[
                ["Soprano is an extremely lightweight text to speech model designed to produce highly realistic speech at unprecedented speed."],
                ["Hello! Welcome to Soprano text to speech. This is a short example."],
                ["The quick brown fox jumps over the lazy dog. This sentence contains all letters of the alphabet."],
                ["Artificial intelligence is transforming the world in ways we never imagined. It's revolutionizing industries and changing how we interact with technology."],
                ["In a distant future, humanity has colonized the stars. Advanced AI systems govern interstellar travel, ensuring safety and efficiency across vast cosmic distances. Explorers venture into uncharted territories, seeking new worlds and civilizations."],
                ["To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles And by opposing end them. To die—to sleep, no more; and by a sleep to say we end The heart-ache and the thousand natural shocks That flesh is heir to: 'tis a consummation Devoutly to be wish'd. To die, to sleep; To sleep, perchance to dream—ay, there's the rub: For in that sleep of death what dreams may come, When we have shuffled off this mortal coil, Must give us pause—there's the respect That makes calamity of so long life."],
            ],
            inputs=[text_input],
            label="Examples",
        )

        async def check_and_set_active_generate(active_func, *args):
            if active_func is not None and active_func != "ready":
                return None, f"Error: Please press Clear first. Current operation: {active_func}", active_func
            result = await generate_speech(args[0], args[1], args[2], args[3])
            return result[0], result[1], "generate"

        async def check_and_set_active_speak(active_func, *args):
            if active_func is not None and active_func != "ready":
                return f"Error: Please press Clear first. Current operation: {active_func}", active_func
            result = await speak_realtime(args[0], args[1], args[2], args[3])
            return result, "speak"

        def clear_active_state():
            return "ready"

        def clear_inputs(active_func):
            with stream_lock:
                global current_stream, current_pyaudio_instance
                if current_stream:
                    try:
                        if hasattr(current_stream, 'is_active') and current_stream.is_active():
                            current_stream.stop_stream()
                        current_stream.close()
                    except Exception:
                        pass
                    current_stream = None
                if current_pyaudio_instance:
                    try:
                        current_pyaudio_instance.terminate()
                    except Exception:
                        pass
                    current_pyaudio_instance = None
            return "", None, "Ready for input...", "ready"

        generate_btn.click(
            fn=check_and_set_active_generate,
            inputs=[active_function, text_input, temperature, top_p, repetition_penalty],
            outputs=[audio_output, status_output, active_function]
        )

        speak_btn.click(
            fn=check_and_set_active_speak,
            inputs=[active_function, text_input, temperature, top_p, repetition_penalty],
            outputs=[status_output, active_function]
        )

        clear_btn.click(
            fn=clear_inputs,
            inputs=[active_function],
            outputs=[text_input, audio_output, status_output, active_function]
        )

        gr.Markdown(
            """

    <br>

    ### Usage tips:

    - Soprano works best when each sentence is between 2 and 15 seconds long.
    - Although Soprano recognizes numbers and some special characters, it occasionally mispronounces them.
      Best results can be achieved by converting these into their phonetic form.
      (1+1 -> one plus one, etc)
    - If Soprano produces unsatisfactory results, you can easily regenerate it for a new, potentially better generation.
      You may also change the sampling settings for more varied results.
    - Avoid improper grammar such as not using contractions, multiple spaces, etc.
    """
        )

    return demo


def find_free_port(start_port=7860, max_tries=100):
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise OSError("Could not find a free port")


def main():
    global current_stream, current_pyaudio_instance

    port = find_free_port(7860)
    print(f"Starting Gradio interface on port {port}")

    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            print("Set Windows Proactor event loop policy")
        except Exception as e:
            print(f"Could not set Windows Proactor event loop policy: {e}")

    demo = create_gradio_interface()

    try:
        demo.queue(max_size=20).launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            theme=gr.themes.Soft(primary_hue="green"),
            prevent_thread_lock=False,
            show_error=True,
            quiet=False,
            favicon_path=None,
            ssl_verify=False,
            max_threads=40,
            css="""
a {
    color: var(--primary-600);
}
a:hover {
    color: var(--primary-700);
}
""",
            root_path=""
        )
        print("Gradio interface launched successfully")
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        with stream_lock:
            if current_stream:
                try:
                    if hasattr(current_stream, 'is_active') and current_stream.is_active():
                        current_stream.stop_stream()
                    current_stream.close()
                except Exception:
                    pass
            if current_pyaudio_instance:
                try:
                    current_pyaudio_instance.terminate()
                except Exception:
                    pass
        sys.exit(0)
    except Exception as e:
        print(f"Error starting Gradio interface: {e}")
        traceback.print_exc()
        with stream_lock:
            if current_stream:
                try:
                    if hasattr(current_stream, 'is_active') and current_stream.is_active():
                        current_stream.stop_stream()
                    current_stream.close()
                except Exception:
                    pass
            if current_pyaudio_instance:
                try:
                    current_pyaudio_instance.terminate()
                except Exception:
                    pass
        sys.exit(1)

if __name__ == "__main__":
    main()
