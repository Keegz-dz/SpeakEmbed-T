from scripts.main import Main
import soundfile as sf
import librosa


def main():
    audio_path = r"Male.mp3"
    wav, sample_rate = librosa.load(audio_path, sr=16000)

    vc = Main()
    # Always give a text argument or else it will default to Whisper which might break the code on some Windows devices.
    intended_text_2 = "Hello everyone, it's nice to speak with you today. This is a demonstration of voice cloning technology."

    wav_out = vc.clone_audio(wav, use_vocoder=True, text=intended_text_2)
    sf.write("intended_text_3.wav", wav_out, sample_rate)


if __name__ == "__main__":
    main()

