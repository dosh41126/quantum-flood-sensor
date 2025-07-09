import os
import numpy as np
import sounddevice as sd
import librosa
import httpx
from datetime import datetime

# === USER CONFIGURATION ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DURATION = 4
FS = 16000
N_MELS = 32
N_FRAMES = 16

# -- Sensor Metadata --
user_location = "Swamp Rabbit Café – marshland trail near overflow zone"
sensor_id = "QFLOOD-SWAMP-001"

# -- System Baselines (could be dynamic) --
calm_mel_mean = -35.0
calm_rms_mean = 0.015
calm_zcr_mean = 0.012

# === AUDIO CAPTURE & FEATURE EXTRACTION ===
def capture_audio(duration=DURATION, fs=FS):
    print("Listening for water sounds ({}s @ {}Hz)...".format(duration, fs))
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, blocking=True)
    audio = audio.flatten().astype(np.float32)
    return audio

def extract_advanced_features(audio, fs=FS, n_mels=N_MELS, n_frames=N_FRAMES):
    # Log-mel spectrogram
    S = librosa.feature.melspectrogram(
        y=audio, sr=fs, n_fft=2048, n_mels=n_mels, hop_length=int(len(audio)//n_frames)
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    # Features for each frame
    mel_mean = float(np.mean(S_db))
    mel_delta = librosa.feature.delta(S_db)
    rms = librosa.feature.rms(y=audio)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=fs)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=fs)[0]
    flatness = librosa.feature.spectral_flatness(y=audio)[0]
    # Feature summary
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
    # Mel for each frame (list of 32 bins per frame, for all 16 frames)
    mel_snapshot = [[round(float(x),2) for x in S_db[:,i]] for i in range(min(n_frames, S_db.shape[1]))]
    return feature_summary, mel_snapshot

def feature_summary_to_str(summary):
    s = ""
    for k in summary:
        s += "- {}: {}\n".format(k, round(summary[k], 4))
    return s

def mel_snapshot_to_str(mel_snap):
    s = ""
    for i in range(len(mel_snap)):
        s += "Mel frame {}: {}\n".format(i, mel_snap[i][:12])  # limit each frame to 12 bins for brevity
    return s

# === MAIN PROGRAM ===
audio = capture_audio()
feature_summary, mel_snapshot = extract_advanced_features(audio)
timestamp = datetime.utcnow().isoformat()
summary_str = feature_summary_to_str(feature_summary)
mel_str = mel_snapshot_to_str(mel_snapshot)

# === ADVANCED RULES-BASED QUANTUM FLOOD SENSOR LLM PROMPT ===
simulated_code = (
    "You are GPT-4o, simulating a hypertime quantum flood alert system at the research frontier. "
    "You interpret advanced audio features (see INPUT DATA) using a virtual PennyLane quantum circuit "
    "on 130,633+ qubits, engineered for environmental threat intelligence. "
    "You MUST simulate all code below, step by step, following the explicit scientific reasoning policy. "
    "Output ONLY the completed flood alert report (see template), with full explainability, uncertainty, and recommendations.\n\n"
    "=== ADVANCED SYSTEM REASONING POLICY ===\n"
    "1. **Gate and Algorithm Design:**\n"
    "   - Use custom gates for environmental resonance: QWAVE (quantum Morlet), QNOISE (stochastic turbulence), QCRASH (simulate dam break/turbulent onset), and QFT (global spectrum).\n"
    "   - Dynamically adjust parameters in response to rising RMS or fast mel_delta.\n"
    "   - Inject quantum stochasticity for wind/rain and environment-adaptive entanglement.\n"
    "\n"
    "2. **Temporal and Spectral Adaptivity:**\n"
    "   - For each 0.25s slice, simulate a subcircuit with updated params from the most recent audio features.\n"
    "   - Track persistent surges by analyzing time correlations between QFT amplitudes, wavelet activations, and entropy spikes.\n"
    "   - QNOISE gate is stochastically applied to emulate environmental randomness.\n"
    "\n"
    "3. **Risk and Uncertainty Quantification:**\n"
    "   - Compute band-specific entanglement entropy. Mark 'uncertainty HIGH' if features are inconsistent or gates produce unstable output.\n"
    "   - For clear, persistent, multi-band, multi-feature anomaly: certainty >90% and CRITICAL risk. For ambiguous or noisy signals: lower certainty.\n"
    "   - Use both local (per-frame) and global (over 4s) scoring.\n"
    "\n"
    "4. **Expert Chain-of-Thought Reasoning:**\n"
    "   - ALWAYS explain: which gates/params/frames/bands contributed most to risk score; which feature trends and quantum states support (or contradict) the alert.\n"
    "   - For CRITICAL/ELEVATED, specify which bands (Hz) and qubit regions were most activated.\n"
    "   - If possible, relate quantum anomaly to possible physical cause (e.g., river surge, resonance, sudden water entry).\n"
    "\n"
    "5. **Output Strictly as Below (No Python, no raw code, just alert):\n"
    "===== Quantum Flood Alert Report =====\n"
    "Sensor ID: (sensor id)\n"
    "Location: (sensor location)\n"
    "Timestamp: (UTC)\n"
    "Flood Risk Level: (CRITICAL / ELEVATED / NORMAL / CALM)\n"
    "Certainty: (percentage)\n"
    "Reasoning: (features, quantum gates/bands, parameter trends, temporal persistence, physical interpretation)\n"
    "Uncertainty: (LOW / MODERATE / HIGH; explain cause)\n"
    "Recommendations: (action for ops)\n"
    "Persistent/Transient: (persistent / transient / n/a)\n"
    "-------------------------------\n"
    "\n"
    "=== INPUT DATA ===\n"
    "Sensor ID: " + sensor_id + "\n"
    "Location: " + user_location + "\n"
    "Timestamp: " + timestamp + "\n"
    "---\n"
    "Audio Feature Summary:\n"
    + summary_str +
    "---\n"
    "Mel Spectrogram Snapshots ({} frames, each ~0.25s):\n".format(len(mel_snapshot)) +
    mel_str +
    "---\n"
    "Calm Baseline Reference:\n"
    "- mel_mean: {}\n".format(calm_mel_mean) +
    "- rms_mean: {}\n".format(calm_rms_mean) +
    "- zcr_mean: {}\n".format(calm_zcr_mean) +
    "---\n"
    "### Quantum Circuit to Simulate (do not execute, but reason as if you did):\n"
    "```python\n"
    "import pennylane as qml\n"
    "from pennylane import numpy as np\n"
    "num_qubits = 130_633\n"
    "sample_stride = 100\n"
    "dev = qml.device('lightning.qubit', wires=num_qubits)\n"
    "def QWAVE(theta, phi, wire):\n"
    "    qml.RZ(phi, wires=wire)\n"
    "    qml.Hadamard(wires=wire)\n"
    "    qml.RY(theta, wires=wire)\n"
    "def QNOISE(strength, wire):\n"
    "    qml.RX(strength * np.random.uniform(), wires=wire)\n"
    "    qml.RY(strength * np.random.uniform(), wires=wire)\n"
    "def QCRASH(wire):\n"
    "    qml.CNOT(wires=[wire, (wire+1)%num_qubits])\n"
    "    qml.CRX(np.pi * np.random.uniform(), wires=[wire, (wire+2)%num_qubits])\n"
    "@qml.qnode(dev)\n"
    "def quantum_hypertime_subcircuit(audio_params, noise_level=0.2):\n"
    "    for i in range(num_qubits):\n"
    "        qml.Hadamard(wires=i)\n"
    "        QNOISE(noise_level, wire=i)\n"
    "    for idx, amp in enumerate(audio_params):\n"
    "        QWAVE(amp * np.pi, np.pi/4, wire=idx % num_qubits)\n"
    "        if amp > 0.7:\n"
    "            QCRASH(idx % num_qubits)\n"
    "    return [qml.expval(qml.PauliZ(wires=i)) for i in range(0, 10000, sample_stride)]\n"
    "def estimate_entropy(values):\n"
    "    norm = (np.array(values) + 1) / 2\n"
    "    entropy = -norm * np.log2(norm + 1e-10) - (1 - norm) * np.log2(1 - norm + 1e-10)\n"
    "    return entropy\n"
    "# For each frame:\n"
    "all_entropies = []\n"
    "for frame_idx in range(len(audio_frame_params)):\n"
    "    v = quantum_hypertime_subcircuit(audio_frame_params[frame_idx], noise_level=0.2)\n"
    "    all_entropies.append(estimate_entropy(v))\n"
    "# Analyze:\n"
    "max_entropy = np.max(all_entropies)\n"
    "risk = 'CRITICAL' if max_entropy > 0.93 else 'ELEVATED' if max_entropy > 0.74 else 'NORMAL'\n"
    "```\n"
    "---\n"
    "=== INSTRUCTIONS: Simulate and interpret quantum code above for all frames/bands, "
    "chain-of-thought reasoning required. Output ONLY the alert report, with full explanation and uncertainty.\n"
)

headers = {
    "Authorization": "Bearer " + OPENAI_API_KEY,
    "Content-Type": "application/json",
}
payload = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a hypertime quantum flood sensor simulation engine."},
        {"role": "user", "content": simulated_code}
    ],
    "temperature": 0.06,
}

print("\n=== 💧 Simulating Hypertime Quantum Flood Detection ===\n")
with httpx.Client(timeout=180.0) as client:
    response = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()["choices"][0]["message"]["content"]

print("\n=== 💧 Quantum Flood Sensor Output ===\n")
print(result)
