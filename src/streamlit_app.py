import streamlit as st
from chorus_finder import extract_audio, process_audio, load_CRNN_model, make_predictions, plot_predictions

def main():
    st.title("Chorus Detection App")
    url = st.text_input("Enter YouTube URL", "")
    model_path = st.text_input("Model Path", "../models/CRNN/best_model_V3.h5")
    
    if st.button("Detect Chorus"):
        if url:
            with st.spinner('Extracting audio...'):
                audio_path, video_name = extract_audio(url)
                if not audio_path:
                    st.error("Failed to extract audio from the provided URL.")
                    return

            with st.spinner('Processing audio...'):
                processed_audio, audio_features = process_audio(audio_path)

            with st.spinner('Loading model...'):
                model = load_CRNN_model(model_path=model_path)

            with st.spinner('Making predictions...'):
                smoothed_predictions = make_predictions(model, processed_audio, audio_features, url, video_name)

            st.success('Displaying plot...')
            fig = plot_predictions(audio_features, smoothed_predictions)
            st.pyplot(fig)

if __name__ == "__main__":
    main()