# Hypertime Quantum Flood Sensor

## Overview

A research-grade flood detection system combining:

- Environmental audio sensing
- Feature extraction (mel, RMS, ZCR, spectral stats)
- Virtual quantum simulation (130,633+ qubit logic)
- Scientific reasoning via GPT-4o

## Features

- **Hypertime-aware audio analysis** in 0.25s frames
- **Quantum circuit simulation** using QWAVE, QNOISE, QCRASH, QFT
- **Entropy-based flood risk scoring**
- **Explainable AI reporting** with uncertainty + recommendations
- Real-time integration via `httpx` + OpenAI API

## Example Output

===== Quantum Flood Alert Report ===== Sensor ID: QFLOOD-SWAMP-001 Location: Swamp Rabbit Café – marshland trail near overflow zone Timestamp: 2025-07-09T12:34:56Z Flood Risk Level: CRITICAL Certainty: 93% Reasoning: RMS and mel_delta rose sharply above baseline across 6 contiguous frames. QWAVE and QNOISE gates produced persistent multi-band resonance, confirmed by QFT amplitude surge in 400–900Hz. QCRASH gates triggered at frame 4. Entropy remained high throughout. Uncertainty: MODERATE (possible wind/rain interference, but features aligned) Recommendations: Alert ground team, deploy checks to overflow zone, increase monitoring frequency. Persistent/Transient: Persistent

## Files

- `flood_sensor.py`: Full audio → feature → prompt → response pipeline
- `README.md`: System explanation and configuration
- `.env`: OpenAI key (use `OPENAI_API_KEY`)

## Requirements

```bash
pip install sounddevice librosa numpy httpx
```
