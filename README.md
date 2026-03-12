# 🏒 IceBreaker — Pro Hockey Success Predictor

## 🌟 Overview

**IceBreaker** is a machine learning pipeline designed to predict the probability of an athlete achieving professional-level success in hockey. By analyzing a rich combination of physical performance metrics, psychological scores, demographic background, and training history, the model generates a personalized **pro-success probability** and compares the athlete's profile against known professional players using **cosine similarity**.

This tool is built for coaches, scouts, and sports scientists who want data-driven insights into athlete development potential.

---

## 🧠 How It Works

The system loads a pre-trained **logistic regression pipeline** (`pro_success_pipeline.joblib`) that was trained on a dataset of hockey athletes across multiple competition categories. Given a new athlete's profile, it:

1. 🔢 **Imputes** any missing physical measurements using training-set medians
2. ⚖️ **Scales** all features using the same scaler fitted during training
3. 🎯 **Predicts** the probability of pro success (0–100%)
4. 📐 **Computes cosine similarity** between the athlete and the top 15 known professional athletes in the training set
5. 📊 **Visualizes** the results with clean, interpretable charts

---

## 📁 Project Structure

| File | Description |
|---|---|
| `IceBreaker.py` | 🚀 Main entry point — define an athlete profile and run predictions |
| `penguins_predictor.py` | 🧩 Core prediction engine — handles loading, inference, and visualization |
| `prediction.py` | 🔬 Standalone script version for quick single-athlete prediction |
| `pro_success_pipeline.joblib` | 🤖 Pre-trained model pipeline (scaler, imputer, logistic regression, pro athlete embeddings) |
| `log_reg_model.ipynb` | 📓 Jupyter notebook used to train and evaluate the model |
| `libr.txt` | 📦 Required Python library versions |
| `synthetic_athletes_full.csv` | 🗂️ Full synthetic athlete dataset used for training/testing |
| `mvp_data.csv` / `final.csv` | 📋 Supporting datasets |

---

## 📊 Features Used by the Model

The model draws on **four major categories** of athlete data:

🏃 **Physical Performance** — speed, power, agility, endurance (e.g., 10-yard dash, broad jump, vertical jump, shuttle times, pull-ups, wattage output)

🧠 **Psychological Scores** — mental toughness, grit, resilience, self-efficacy, sport motivation, teamwork, pre-game anxiety

🎓 **Demographic & Socioeconomic** — parent education, household income, financial support for hockey, access to sports facilities

🏑 **Hockey History & Training Load** — age of first league play, weekly practice/game hours, camp frequency, competition level, multi-sport involvement

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r libr.txt
```

### 2. Run a prediction

Edit the athlete profile dictionary inside `IceBreaker.py` with your athlete's data, then run:
```bash
python IceBreaker.py
```

### 3. Output

The program will print the predicted pro-success probability, cosine similarity to known pros, and display two visualizations:
- 📊 A score chart showing the probability and similarity side by side
- 📈 A feature importance chart showing the top 15 factors driving the prediction

---

## 📦 Requirements
```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
matplotlib==3.9.2
seaborn==0.13.2
scipy==1.14.1
```

---

## ⚠️ Notes

- The `.joblib` file is **required** for the program to run — it contains the trained model, scaler, imputer, and pro athlete reference embeddings. Without it, the pipeline cannot make predictions.
- The model was trained on a **synthetic dataset** designed to reflect realistic hockey athlete distributions. Results should be interpreted as exploratory insights, not definitive evaluations.
- Some physical measurements (e.g., `shuttle_100yd`, `pull_ups`, `ue_watt`) may be `None` if not measured — the pipeline handles these gracefully via median imputation.

---

## 🤝 Contributing

Pull requests are welcome! If you'd like to improve the model, add new features, or build a UI on top of the prediction engine, feel free to fork and contribute.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
