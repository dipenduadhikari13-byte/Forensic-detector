# 🧠 Forensic Detector (FastAPI + Streamlit)

A cloud-deployable, CPU-compatible image forensics system for detecting **AI-generated and manipulated images** using multi-layer analysis.

> ⚠️ No forensic system guarantees 100% accuracy. This tool is designed for **robust, explainable, real-world detection**, not absolute certainty.

---

## 🚀 Why This Matters
With the rise of diffusion models and advanced editing tools, distinguishing real vs manipulated images is becoming increasingly difficult.

This project combines **classical digital forensics + AI-based detection** to provide:
- practical reliability  
- explainable outputs  
- deployable architecture  

---

## ✨ Features

### 🔍 Detection Capabilities
- AI-generated image detection (diffusion / GAN)
- Image manipulation detection:
  - Splicing
  - Copy-move
  - Inpainting
- Signal-level forensic analysis:
  - ELA (Error Level Analysis)
  - Noise inconsistency
  - FFT frequency artifacts

---

### ⚙️ System Features
- Async FastAPI backend
- Lazy model loading (fast startup)
- CPU-compatible (no GPU required)
- Fail-safe fallback (works without model checkpoint)
- Streamlit UI for quick interaction
- Heatmap visualization (optional)

---

## 📂 Project Structure