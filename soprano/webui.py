#!/usr/bin/env python3
"""
Soprano TTS Web UI
"""
import gradio as gr
import torch
from soprano import SopranoTTS


def create_app():
    # Initialize the TTS model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tts = SopranoTTS(device=device, cache_size_mb=100)
    
    def synthesize(text, temperature, top_p, repetition_penalty):
        if not text.strip():
            return "Error: Text cannot be empty", None
        
        # Generate audio
        audio_tensor = tts.infer(
            text=text,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        
        # Convert tensor to numpy for Gradio
        audio_np = audio_tensor.cpu().numpy()
        
        # Return the audio as a tuple (sample_rate, audio_data)
        return "Audio generated successfully!", (32000, audio_np)

    # Create Gradio interface
    with gr.Blocks(title="Soprano TTS Web UI") as demo:
        gr.Markdown("# 🎵 Soprano TTS Web UI")
        gr.Markdown("Convert text to realistic speech using Soprano TTS")
        
        with gr.Row():
            with gr.Column():
                text_input = gr.TextArea(label="Input Text", placeholder="Enter text to synthesize...", elem_id="text_input")
                
                with gr.Group():
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
                
                generate_btn = gr.Button("Generate Speech", variant="primary")
            
            with gr.Column():
                status_output = gr.Textbox(label="Status", interactive=False)
                audio_output = gr.Audio(label="Generated Speech", type="numpy")
        
        generate_btn.click(
            fn=synthesize,
            inputs=[text_input, temperature, top_p, repetition_penalty],
            outputs=[status_output, audio_output]
        )
        
        gr.Examples(
            examples=[
                ["Hello, welcome to Soprano TTS. This is a demonstration of the web interface."],
                ["The quick brown fox jumps over the lazy dog."],
                ["Soprano is an extremely lightweight text to speech model that can achieve high quality audio synthesis."],
            ],
            inputs=[text_input],
        )
    
    return demo


def main():
    app = create_app()
    print("Starting Soprano TTS Web UI...")
    print("Visit http://localhost:7860 to access the interface")
    app.launch(server_name="localhost", server_port=7860)


if __name__ == "__main__":
    main()