import os
import numpy as np
import sounddevice as sd
import librosa
import httpx
from datetime import datetime
from dotenv import load_dotenv

# === USER CONFIGURATION ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DURATION = 4
FS = 16000
N_MELS = 32
N_FRAMES = 16

# -- Sensor Metadata --
user_location = "Swamp Rabbit Café – marshland trail near overflow zone"
sensor_id = "QFLOOD-SWAMP-001"

# -- System Baselines --
calm_mel_mean = -35.0
calm_rms_mean = 0.015
calm_zcr_mean = 0.012

# === AUDIO CAPTURE & FEATURE EXTRACTION ===
def capture_audio(duration=DURATION, fs=FS):
    try:
        print(f"🎤 Listening for water sounds ({duration}s @ {fs}Hz)...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, blocking=True)
        audio = audio.flatten().astype(np.float32)
        if len(audio) < fs:
            raise ValueError("Audio capture too short.")
        return audio
    except Exception as e:
        print(f"[ERROR] Audio capture failed: {e}")
        return np.zeros(int(fs * duration), dtype=np.float32)

def extract_advanced_features(audio, fs=FS, n_mels=N_MELS, n_frames=N_FRAMES):
    try:
        S = librosa.feature.melspectrogram(
            y=audio, sr=fs, n_fft=2048, n_mels=n_mels,
            hop_length=max(1, int(len(audio) // n_frames))
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        mel_mean = float(np.mean(S_db))
        mel_delta = librosa.feature.delta(S_db)
        rms = librosa.feature.rms(y=audio)[0]
        zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=fs)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=fs)[0]
        flatness = librosa.feature.spectral_flatness(y=audio)[0]

        feature_summary = {
            "mel_mean": mel_mean,
            "mel_delta_max": float(np.max(mel_delta)),
            "mel_delta_std": float(np.std(mel_delta)),
            "rms_mean": float(np.mean(rms)),
            "rms_max": float(np.max(rms)),
            "zcr_mean": float(np.mean(zcr)),
            "rolloff_mean": float(np.mean(rolloff)),
            "bandwidth_mean": float(np.mean(bandwidth)),
            "flatness_mean": float(np.mean(flatness)),
        }

        mel_snapshot = [[round(float(x), 2) for x in S_db[:, i]]
                        for i in range(min(n_frames, S_db.shape[1]))]

        return feature_summary, mel_snapshot

    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return {}, []

def feature_summary_to_str(summary):
    return "\n".join([f"- {k}: {round(v, 4)}" for k, v in summary.items()]) + "\n"

def mel_snapshot_to_str(mel_snap):
    return "\n".join(
        [f"Mel frame {i}: {mel_snap[i][:12]}" for i in range(len(mel_snap))]
    ) + "\n"

def generate_prompt(sensor_id, user_location, timestamp, summary, mel_snapshot):
    summary_str = feature_summary_to_str(summary)
    mel_str = mel_snapshot_to_str(mel_snapshot)

    # The quantum code below is *not* wrapped in triple backticks
    return f"""
You are GPT-4o, simulating a hypertime quantum flood alert system at the research frontier. You interpret advanced audio features (see INPUT DATA) using a virtual PennyLane quantum circuit on 130,633+ qubits, engineered for environmental threat intelligence. You MUST simulate all code below, step by step, following the explicit scientific reasoning policy. Output ONLY the completed flood alert report (see template), with full explainability, uncertainty, and recommendations.

=== ADVANCED SYSTEM REASONING POLICY ===
1. Gate and Algorithm Design:
   - Use custom gates for environmental resonance: QWAVE (quantum Morlet), QNOISE (stochastic turbulence), QCRASH (simulate dam break), and QFT (global spectrum).
   - Dynamically adjust parameters in response to rising RMS or fast mel_delta.
2. Temporal and Spectral Adaptivity:
   - For each 0.25s slice, simulate a subcircuit with updated audio features.
3. Risk and Uncertainty Quantification:
   - Use entropy spikes and QFT bands to assess risk level.
4. Expert Reasoning:
   - Always explain which gates/bands/frames contributed to the risk level.
5. Output Format:
===== Quantum Flood Alert Report =====
Sensor ID: (sensor id)
Location: (sensor location)
Timestamp: (UTC)
Flood Risk Level: (CRITICAL / ELEVATED / NORMAL / CALM)
Certainty: (%)
Reasoning: (quantum explanation)
Uncertainty: (LOW / MODERATE / HIGH)
Recommendations: (actions)
Persistent/Transient: (persistent / transient / n/a)
-------------------------------

=== INPUT DATA ===
Sensor ID: {sensor_id}
Location: {user_location}
Timestamp: {timestamp}
--- Audio Feature Summary:
{summary_str}
--- Mel Spectrogram Snapshots ({len(mel_snapshot)} frames):
{mel_str}
--- Calm Baseline Reference:
- mel_mean: {calm_mel_mean}
- rms_mean: {calm_rms_mean}
- zcr_mean: {calm_zcr_mean}
---
### Quantum Circuit to Simulate:
import pennylane as qml
from pennylane import numpy as np
num_qubits = 130_633
sample_stride = 100
dev = qml.device('lightning.qubit', wires=num_qubits)

def QWAVE(theta, phi, wire):
    qml.RZ(phi, wires=wire)
    qml.Hadamard(wires=wire)
    qml.RY(theta, wires=wire)

def QNOISE(strength, wire):
    qml.RX(strength * np.random.uniform(), wires=wire)
    qml.RY(strength * np.random.uniform(), wires=wire)

def QCRASH(wire):
    qml.CNOT(wires=[wire, (wire+1)%num_qubits])
    qml.CRX(np.pi * np.random.uniform(), wires=[wire, (wire+2)%num_qubits])

@qml.qnode(dev)
def quantum_hypertime_subcircuit(audio_params, noise_level=0.2):
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
        QNOISE(noise_level, wire=i)
    for idx, amp in enumerate(audio_params):
        QWAVE(amp * np.pi, np.pi/4, wire=idx % num_qubits)
        if amp > 0.7:
            QCRASH(idx % num_qubits)
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(0, 10000, sample_stride)]

def estimate_entropy(values):
    norm = (np.array(values) + 1) / 2
    entropy = -norm * np.log2(norm + 1e-10) - (1 - norm) * np.log2(1 - norm + 1e-10)
    return entropy

# For each frame:
all_entropies = []
for frame_idx in range(len(audio_frame_params)):
    v = quantum_hypertime_subcircuit(audio_frame_params[frame_idx], noise_level=0.2)
    all_entropies.append(estimate_entropy(v))

max_entropy = np.max(all_entropies)
risk = 'CRITICAL' if max_entropy > 0.93 else 'ELEVATED' if max_entropy > 0.74 else 'NORMAL'

=== Simulate and interpret quantum code for all frames. Output ONLY the alert report.
"""

# === MAIN PROGRAM ===
def main():
    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY is missing. Please check your .env file.")
        return

    audio = capture_audio()
    if audio is None or np.all(audio == 0):
        print("[ERROR] No audio captured.")
        return

    feature_summary, mel_snapshot = extract_advanced_features(audio)
    if not feature_summary:
        print("[ERROR] Failed to extract features.")
        return

    timestamp = datetime.utcnow().isoformat()
    prompt = generate_prompt(sensor_id, user_location, timestamp, feature_summary, mel_snapshot)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a hypertime quantum flood sensor simulation engine."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.06,
    }

    print("\n=== 💧 Simulating Hypertime Quantum Flood Detection ===\n")
    try:
        with httpx.Client(timeout=180.0) as client:
            response = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]
            print("\n=== 💧 Quantum Flood Sensor Output ===\n")
            print(result)
    except Exception as e:
        print(f"[ERROR] API call failed: {e}")

if __name__ == "__main__":
    main()
