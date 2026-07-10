"""
Streamlit web application for chorus detection.
"""

import os
import tempfile
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import numpy as np
import soundfile as sf
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time
import sys

from pytorch_core.audio_processor import process_audio
from pytorch_core.model import load_CRNN_model, make_predictions, MODEL_PATH
from pytorch_core.visualization import plot_predictions, plot_chorus_segments
from pytorch_core.utils import extract_audio, cleanup_temp_files


def create_download_link(fig, filename="plot.png"):
    """Convert a matplotlib figure to a PNG image and create a download link."""
    buf = BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_figure(buf, format="png", dpi=150)
    buf.seek(0)
    return buf


def display_audio_player(audio_path, start_time=None, end_time=None):
    """Display an audio player for the full file or a specific segment."""
    try:
        # Load the audio file
        if start_time is not None and end_time is not None:
            # Load the segment and serve it as an in-memory WAV
            y, sr = librosa.load(audio_path, sr=None, offset=start_time, duration=end_time-start_time)
            buf = BytesIO()
            sf.write(buf, y, sr, format="WAV")
            st.audio(buf.getvalue(), format="audio/wav")
        else:
            # Play the full file
            st.audio(audio_path)
    except Exception as e:
        st.error(f"Error playing audio: {e}")


def process_youtube_url(youtube_url):
    """Process a YouTube URL and return the audio path."""
    with st.spinner("Requesting audio from YouTube..."):
        try:
            audio_path, video_title = extract_audio(youtube_url)
        except Exception:
            audio_path, video_title = None, None
        if not audio_path:
            st.error(
                "Could not download audio from this YouTube URL. YouTube "
                "sometimes blocks downloads from hosted servers, or the video "
                "may be age-, region-, or copyright-restricted. Please try "
                "uploading an audio file instead."
            )
            return None, None
        return audio_path, video_title


def process_uploaded_file(uploaded_file):
    """Save an uploaded file to disk and return the path."""
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        audio_path = tmp_file.name
    return audio_path, uploaded_file.name


def analyze_audio(audio_path, video_title=None):
    """Process the audio file and detect choruses."""
    try:
        # Process audio
        with st.spinner("Processing audio..."):
            processed_audio, audio_features = process_audio(audio_path)
            if processed_audio is None:
                st.error("Failed to process audio. Please try a different file.")
                return
        
        # Load model
        with st.spinner("Loading model..."):
            model = load_CRNN_model()
            if model is None:
                st.error("Failed to load model.")
                return
        
        # Make predictions
        with st.spinner("Detecting choruses..."):
            smoothed_predictions, chorus_start_times, chorus_end_times = make_predictions(
                model, processed_audio, audio_features)
        
        # Display results
        st.subheader("Analysis Results")
        
        if len(chorus_start_times) == 0:
            st.warning("No choruses detected in this audio file.")
            
            # Still show the waveform
            fig = plot_predictions(audio_features, smoothed_predictions, 
                                   title=f"No Choruses Detected in {video_title or 'Audio'}")
            st.pyplot(fig)
            
            # Create download link for the plot
            plot_data = create_download_link(fig, "waveform_plot.png")
            st.download_button(
                label="Download Waveform Plot", 
                data=plot_data, 
                file_name="waveform_plot.png", 
                mime="image/png"
            )
            
        else:
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Waveform", "Timeline", "Playback"])
            
            with tab1:
                # Plot the waveform with chorus predictions
                fig1 = plot_predictions(audio_features, smoothed_predictions, 
                                        title=f"Chorus Predictions for {video_title or 'Audio'}")
                st.pyplot(fig1)
                
                # Create download link for the waveform plot
                waveform_plot_data = create_download_link(fig1, "waveform_plot.png")
                st.download_button(
                    label="Download Waveform Plot", 
                    data=waveform_plot_data, 
                    file_name="waveform_plot.png", 
                    mime="image/png"
                )
            
            with tab2:
                # Plot the chorus timeline
                fig2 = plot_chorus_segments(audio_features, chorus_start_times, chorus_end_times, 
                                            title=f"Chorus Timeline for {video_title or 'Audio'}")
                st.pyplot(fig2)
                
                # Create download link for the timeline plot
                timeline_plot_data = create_download_link(fig2, "timeline_plot.png")
                st.download_button(
                    label="Download Timeline Plot", 
                    data=timeline_plot_data, 
                    file_name="timeline_plot.png", 
                    mime="image/png"
                )
            
            with tab3:
                # Display audio players for each chorus
                st.subheader("Full Audio")
                display_audio_player(audio_path)
                
                st.subheader("Chorus Segments")
                for i, (start, end) in enumerate(zip(chorus_start_times, chorus_end_times)):
                    with st.expander(f"Chorus {i+1}: {int(start//60)}:{int(start%60):02d} - {int(end//60)}:{int(end%60):02d}"):
                        display_audio_player(audio_path, start, end)
        
        # Return the results
        return {
            "audio_features": audio_features,
            "predictions": smoothed_predictions,
            "chorus_start_times": chorus_start_times,
            "chorus_end_times": chorus_end_times
        }
        
    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None
    finally:
        # Clean up any temporary YouTube files
        cleanup_temp_files()


def main():
    """Main function for the Streamlit app."""
    st.set_page_config(
        page_title="Automated Chorus Detection",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    div.stButton > button {
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Automated Chorus Detection")
    st.markdown("""
    Upload an audio file or provide a YouTube URL to get started.
    """)
    
    # Create a sidebar for app explanation
    with st.sidebar:
        st.markdown("""
        ## How it works

        The pipeline extracts audio features (RMS energy, key-invariant chroma, mel spectrogram, MFCC, and tempogram activations), segments them by musical meter, and feeds them to a convolutional recurrent neural network that predicts a chorus probability for each meter. See the [GitHub repository](https://github.com/dennisvdang/chorus-detection) for the full architecture, training details, and source code.
        
        ## Supported formats
        
        - MP3
        - WAV
        - FLAC
        - M4A
        - OGG
        
        ## YouTube support

        Paste a YouTube URL to analyze a song directly.

        **Note:** The YouTube option is likely blocked on this server-hosted demo. To use it,
        [install the app locally](https://github.com/dennisvdang/chorus-detection)
        and run it from your own machine.
        """)
        
        st.header("Credits")
        st.markdown("Developed by Dennis Dang")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Audio", "YouTube URL"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload an audio file", 
            type=["mp3", "wav", "flac", "m4a", "ogg"],
            help="Select an audio file from your computer"
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file)
            
            if st.button("Analyze", type="primary", key="analyze_upload"):
                audio_path, file_name = process_uploaded_file(uploaded_file)
                try:
                    analyze_audio(audio_path, file_name)
                finally:
                    # Clean up temp file
                    try:
                        if os.path.exists(audio_path):
                            os.unlink(audio_path)
                    except:
                        pass
    
    with tab2:
        youtube_url = st.text_input(
            "Enter YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste a YouTube URL to analyze"
        )
        
        if youtube_url:
            if st.button("Analyze", type="primary", key="analyze_youtube"):
                audio_path, video_title = process_youtube_url(youtube_url)
                if audio_path:
                    analyze_audio(audio_path, video_title)


if __name__ == "__main__":
    main()
    
# Entry point for command-line execution
def run_web_app():
    """Entry point for the Streamlit web app from command line."""
    print("Starting Chorus Detection Web App...")
    print("Access the app in your browser at http://localhost:8501")
    sys.argv = ["streamlit", "run", __file__]
    import streamlit.web.cli as stcli
    sys.exit(stcli.main()) 