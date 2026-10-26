# Sign Language Translator 🤟

A real-time **Sign Language Translator** built using computer vision and machine learning. The application recognizes predefined sign language gestures through a webcam and converts them into text and spoken audio, helping make communication more accessible.

## ✨ Features

* **Real-Time Gesture Recognition** — Detects and classifies gestures using a webcam.
* **Hand & Pose Tracking** — Uses MediaPipe to extract hand and body landmarks.
* **Machine Learning Classification** — Supports trained Random Forest, SVM, and XGBoost models.
* **Text-to-Speech** — Converts recognized gestures into spoken output.
* **Prediction Stabilization** — Smooths predictions to improve consistency during live recognition.
* **Sign History** — Displays previously recognized gestures.
* **16 Gesture Classes** — Recognizes predefined signs and phrases, including greetings, requests, and common expressions.

## 🛠️ Tech Stack

| Technology     | Purpose                                      |
| -------------- | -------------------------------------------- |
| Python         | Core development and model integration       |
| OpenCV         | Webcam access and real-time video processing |
| MediaPipe      | Hand and body pose landmark detection        |
| Scikit-learn   | Machine learning models and evaluation       |
| XGBoost        | Gradient-boosted classification              |
| NumPy & Pandas | Feature processing and dataset management    |

## ⚙️ How It Works

1. **Video Capture:** OpenCV captures live frames from the webcam.
2. **Landmark Extraction:** MediaPipe detects hand and body pose landmarks.
3. **Feature Processing:** Landmark coordinates are converted into numerical feature vectors.
4. **Gesture Classification:** A trained machine learning model predicts the corresponding gesture.
5. **Output Generation:** The recognized gesture is displayed on screen and converted into speech.

## 🚀 Getting Started

### Prerequisites

* Python 3.10 or a compatible version for your dependencies
* A working webcam
* Git

### Installation

**1. Clone the repository**

```bash
git clone <your-repository-url>
cd <your-project-folder>
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

**3. Install dependencies**

If the repository includes a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the required packages individually based on your project configuration.

**4. Download the MediaPipe model files**

Place the required hand and pose landmark model files in the paths expected by the application. The project uses MediaPipe Hand Landmarker and Pose Landmarker models.

**5. Run the application**

```bash
python webcam.py
```

Make sure your trained model files and configuration files are available at the paths expected by the application.

> **Note:** Adjust the commands and file paths to match your repository structure before publishing this README.

## 📊 Model Evaluation

Three machine learning classifiers were evaluated using a stratified 80/20 training and testing split.

| Model                        | Test Accuracy |
| ---------------------------- | ------------: |
| Support Vector Machine (SVM) |        99.11% |
| Random Forest                |        98.81% |
| XGBoost                      |        98.21% |

These results represent performance on the evaluation dataset. Real-world accuracy may vary depending on lighting, camera positioning, signing speed, and hand visibility.

## 🔮 Future Improvements

* Expand the vocabulary to support more signs and phrases.
* Improve robustness across different users and signing styles.
* Add sentence-level translation and contextual recognition.
* Develop a more accessible interface for everyday use.
