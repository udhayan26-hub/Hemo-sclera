# Hemo-Sclera: Multimodal Diagnostic AI

[![CI Pipeline](https://github.com/udhayan26-hub/Hemo-sclera/actions/workflows/ci.yml/badge.svg)](https://github.com/udhayan26-hub/Hemo-sclera/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg)](pyproject.toml)
[![Next.js Version](https://img.shields.io/badge/Next.js-14.2.5-black.svg)](package.json)
[![Docker Support](https://img.shields.io/badge/Docker-supported-blue.svg)](Dockerfile)

An enterprise-grade multimodal diagnostic triage assistant for non-invasive Anemia (eyelid palpebral conjunctiva) and Jaundice (scleral yellowness) screening.

---

## Table of Contents
- [Overview & Clinical Impact](#overview--clinical-impact)
- [System Architecture](#system-architecture)
- [The Color Science (CIE L*a*b* vs RGB)](#the-color-science-cie-lab-vs-rgb)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Testing Suite](#testing-suite)
- [Docker Deployment](#docker-deployment)
- [CI/CD & Deployment](#cicd--deployment)
- [Contributing](#contributing)
- [License](#license)
- [Security & Disclaimer](#security--disclaimer)

---

## Overview & Clinical Impact

In low-resource settings, access to clinical diagnostic facilities is limited. Hemo-Sclera acts as a high-fidelity front-line triage screening gateway. By utilizing mobile device cameras, it analyzes physiological tissue structures to provide instant risk assessments:
- **Anemia Screening**: Examines palpebral conjunctiva paleness.
- **Jaundice Screening**: Examines sclera yellowness.

This system is designed to act as a supplementary warning system before routing patients to primary care clinics.

---

## System Architecture

Hemo-Sclera is built using a modern, decoupled hybrid stack combining a high-performance Python-based computer vision engine and a premium Next.js web interface:

- **Next.js Web Portal (`pages/index.js`)**: A responsive clinical dashboard for practitioners.
- **Inference API (`pages/api/infer.js`)**: Decoupled serverless route leveraging OpenRouter with multi-model fallback cascade.
- **Vision Engine (`engine/`)**: Implements CIE L*a*b* color science, Hough circle transforms for iris masking, and YOLO-driven tissue segmentation.
- **Streamlit Desktop (`app.py`)**: Local companion interface for offline clinical tests.

---

## The Color Science (CIE L*a*b* vs RGB)

Standard RGB spaces combine color and light intensity, making them highly sensitive to environment illumination. To overcome this, Hemo-Sclera converts images into the **CIE L*a*b*** color space:
- **L* (Lightness)**: Completely isolated to remove lighting/flash variations.
- **a* (Green $\rightarrow$ Red axis)**: Serves as a direct physiological proxy for blood oxygenation and hemoglobin content (Anemia).
- **b* (Blue $\rightarrow$ Yellow axis)**: Serves as a direct physiological proxy for bilirubin accumulation in the sclera (Jaundice).

---

## Installation & Setup

### Python Environment (Vision Engine & Streamlit)
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/udhayan26-hub/Hemo-sclera.git
   cd hemo-sclera
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables in `.env`:
   ```env
   OPENROUTER_API_KEY=your_api_key_here
   ```

### Web Dashboard (Next.js)
1. Install Node.js dependencies:
   ```bash
   npm install
   ```
2. Run the local development server:
   ```bash
   npm run dev
   ```

---

## Usage Guide

### 1. Web Interface
Access the Next.js web application at `http://localhost:3000` to utilize the modern clinical dashboard.

### 2. Streamlit Companion
Run the offline companion tool:
   ```bash
   streamlit run app.py
   ```

### 3. CLI Pipeline Execution
Run discrete tasks using the orchestrator:
   ```bash
   # Run data preparation
   python pipeline.py --task run_data_prep

   # Train proprietary color models
   python pipeline.py --task train_models

   # Predict on patient sample
   python pipeline.py --task predict --image path/to/sample.png --disease jaundice
   ```

---

## Testing Suite

The repository features a fully integrated unit testing suite with pytest:
```bash
# Run all unit tests
pytest
```

---

## Docker Deployment

To build and run the entire ecosystem locally using Docker:
```bash
# Build the Docker image
docker build -t hemo-sclera .

# Run the container
docker run -p 3000:3000 --env-file .env hemo-sclera
```

Or run with docker-compose:
```bash
docker-compose up --build
```

---

## CI/CD & Deployment

- **GitHub Actions**: Automated pipelines run on every push and pull request to ensure linting, testing, and Next.js production building passes.
- **Vercel**: Seamless deployment pipeline for the Next.js clinical dashboard.

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for detailed guidelines.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Security & Disclaimer

This software is for research and screening purposes only. It is **not** a replacement for standard clinical laboratory examinations. Read [SECURITY.md](SECURITY.md) to report vulnerabilities.
