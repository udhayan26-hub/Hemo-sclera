# Contributing to Hemo-Sclera

Thank you for your interest in contributing to Hemo-Sclera. We welcome contributions to improve our diagnostic screening models, pipeline processing, UI dashboards, and documentation.

## How to Contribute

1. **Reporting Bugs**: Open an issue describing the bug, including steps to reproduce, expected behavior, and screenshots or logs.
2. **Suggesting Enhancements**: Open an issue detailing the enhancement, use case, and proposed implementation.
3. **Pull Requests**:
   - Fork the repository and create your branch from `main`.
   - Ensure your code adheres to PEP 8 for Python and standard formatting for JavaScript.
   - Run existing tests and add new tests for your features.
   - Submit a pull request with a clear description of the changes.

## Development Setup

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   npm install
   ```
3. Run unit tests:
   ```bash
   pytest
   ```
4. Start the local servers:
   ```bash
   streamlit run app.py
   npm run dev
   ```
