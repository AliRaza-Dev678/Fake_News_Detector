# 📰 Fake News Detector

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-red?style=flat-square&logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> An NLP-powered web application that classifies news headlines as **Real** or **Fake** in real time — built with a Random Forest classifier, TF-IDF vectorization, and deployed via an interactive Streamlit interface.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Demo](#-demo)
- [Dataset](#-dataset)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🔍 Overview

Misinformation spreads faster than ever in the digital age. This project tackles that problem by building a machine learning pipeline that analyzes a news headline and predicts whether it is genuine or fabricated. The model is trained on a large labeled dataset of real and fake news articles and is served through a clean, user-friendly Streamlit web app.

---

## ❓ Problem Statement

With the explosive growth of social media and online news, fake news has become a serious societal threat — influencing public opinion, elections, and health decisions. The goal of this project is to:

- Automatically classify news headlines as **Real** or **Fake**
- Preprocess and extract meaningful features from raw text using NLP
- Provide an accessible web interface for real-time predictions

---

## 🎬 Demo

1. Launch the app
2. Enter any news headline in the input box
3. Click **"Check News"**
4. Instantly receive a verdict — ✅ **Real** or 🚨 **Fake**

```
Example Input:  "Scientists discover cure for all diseases overnight"
Prediction:     🚨 This news article is Fake!
```

---

## 📊 Dataset

The model is trained on the **WELFake Dataset** — a large-scale benchmark dataset for fake news detection.

| Property | Details |
|---|---|
| **Name** | WELFake Dataset |
| **File** | `WELFake_Dataset.csv` |
| **Size** | ~72,000 news articles |
| **Labels** | `0` = Real, `1` = Fake |
| **Feature Used** | `title` (news headline) |
| **Source** | [Kaggle — WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) |

> ⚠️ The dataset is not included in this repository due to its size. Download it from Kaggle and place it in the root directory as `WELFake_Dataset.csv`.

---

## ⚙️ How It Works

The pipeline follows these steps end-to-end:

### 1. Text Preprocessing
- Remove non-alphabetical characters using regex
- Convert to lowercase
- Tokenize and remove English **stopwords** (via NLTK)
- Apply **Porter Stemming** to reduce words to their root form

### 2. Feature Extraction
- Transform processed text into numerical vectors using **TF-IDF Vectorization** — capturing the importance of each word relative to the entire corpus

### 3. Model Training
- Split data into **80% training / 20% testing** with stratified sampling
- Train a **Random Forest Classifier** with 100 decision trees

### 4. Prediction
- User inputs a headline → same preprocessing pipeline is applied → vectorized → fed to the model → prediction returned

### 5. Web Interface
- Built with **Streamlit** for a fast, interactive user experience

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **Streamlit** | Web application framework |
| **Pandas & NumPy** | Data loading and manipulation |
| **NLTK** | Natural Language Processing (stopwords, stemming) |
| **scikit-learn** | TF-IDF vectorization, train/test split, Random Forest |
| **Regex (`re`)** | Text cleaning and normalization |
| **Jupyter Notebook** | Exploratory analysis and prototyping |

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ and pip installed.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/AliRaza-Dev678/Fake_News_Detector.git
cd Fake_News_Detector

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install required packages
pip install pandas numpy nltk scikit-learn streamlit
```

### Download the Dataset

1. Go to [Kaggle — WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
2. Download `WELFake_Dataset.csv`
3. Place it in the root of the project directory

### Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

### Run the Notebook (Optional)

```bash
jupyter notebook Fake_News_Detector.ipynb
```

---

## 📁 Project Structure

```
Fake_News_Detector/
│
├── app.py                      # Streamlit web application
├── Fake_News_Detector.ipynb    # Exploratory analysis & model prototyping
├── WELFake_Dataset.csv         # Dataset (download separately from Kaggle)
└── README.md                   # Project documentation
```

---

## 📈 Model Performance

The Random Forest model trained on TF-IDF features of news titles achieves strong performance on the WELFake dataset:

| Metric | Score |
|---|---|
| **Accuracy** | ~94–96% |
| **Classifier** | Random Forest (100 trees) |
| **Vectorizer** | TF-IDF on stemmed title tokens |
| **Test Split** | 20% stratified |

> Exact metrics may vary slightly depending on the dataset version used.

---

## 🔭 Future Improvements

- Use **article body text** in addition to the headline for richer context
- Experiment with **Logistic Regression**, **SVM**, and **XGBoost** for comparison
- Integrate **transformer-based models** (e.g., BERT) for state-of-the-art performance
- Add **confidence scores** alongside the prediction
- Deploy publicly via **Streamlit Cloud** or **Hugging Face Spaces**
- Include **explainability** (e.g., LIME or SHAP) to show which words drove the prediction

---

## 👤 Author

**Ali Raza**

[![GitHub](https://img.shields.io/badge/GitHub-AliRaza--Dev678-black?style=flat-square&logo=github)](https://github.com/AliRaza-Dev678)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it.

---

> ⭐ If you found this project useful, consider giving it a star on GitHub!
