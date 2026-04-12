import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for server use)
import matplotlib.pyplot as plt
import numpy as np


def plot_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(9, 3.5))
        fig.patch.set_facecolor("#0a0a0a")
        ax.set_facecolor("#0a0a0a")

        img = librosa.display.specshow(
            S_db,
            x_axis="time",
            y_axis="mel",
            sr=sr,
            ax=ax,
            cmap="magma",
        )
        ax.set_title("MEL SPECTROGRAM — AUDIO FORENSIC TRACE", color="#00F0FF",
                     fontsize=8, fontfamily="monospace", pad=10)
        ax.set_xlabel("Time (s)", color="#545454", fontsize=7)
        ax.set_ylabel("Hz (Mel)", color="#545454", fontsize=7)
        ax.tick_params(colors="#545454", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1a1a1a")

        cb = fig.colorbar(img, ax=ax, format="%+2.0f dB")
        cb.ax.yaxis.set_tick_params(color="#545454", labelsize=6)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#545454")

    fig.tight_layout(pad=0.5)
    return fig