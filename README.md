# Healthcare Claim Prediction System
##📌 Project Overview
A Streamlit-based machine learning app for 🏥 **Healthcare Insurance claim prediction system** built using **Python, Streamlit, OCR, and Scikit-learn**.

This application predicts whether a healthcare insurance claim is likely to be **Approved, At Risk, or Denied**, along with the **denial probability percentage** and clear reasoning based on business rules.

---

## 🚀 Live Demo

🔗 **Streamlit App:**  
https://health-care-claim-denial-predictor-rtfqmpzzwspz6vbh9b4pxa.streamlit.app/

---

## 📌 Features

- 📄 **Prescription Image Upload**
  - Upload prescription images in `.png`, `.jpg`, or `.jpeg`
  - OCR-based text extraction using Tesseract

- 🤖 **Machine Learning Prediction**
  - Predicts denial probability
  - Uses trained ML model with preprocessing

- 📊 **Claim Status Output**
  - `APPROVED`
  - `RISK`
  - `DENIED`

- 📌 **Rule-Based Validation**
  - Out-of-network provider
  - Missing prior authorization
  - High billing amount
  - Late claim submission

- 📁 **Prediction History**
  - Stores previous predictions
  - Download history as CSV report

- 🩺 **Medical Code Mapping**
  - CPT procedure codes
  - ICD-10 diagnosis codes

---

## 🛠 Tech Stack

- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Pickle**
- **Pillow**
- **Pytesseract OCR**

---

## 📂 Project Structure

```text
health-care-claim-denial-predictor/
│
├── app.py
├── model.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
├── packages.txt
└── README.md
```

---

## ⚙️ Installation & Run Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/sushanth0963/Project-files.git
cd Project-files
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application

```bash
streamlit run app.py
```

---

## 📈 How It Works

1. Upload prescription image
2. OCR extracts patient details
3. User enters claim-related information
4. Data is preprocessed and scaled
5. ML model predicts denial probability
6. Rule engine assigns claim status
7. Results + history report are displayed

---

## ⚠️ Disclaimer

This project is developed for **educational and portfolio purposes**.

The prediction results are based on historical claim patterns and rule-based logic.  
Final claim approval decisions should always be manually reviewed by domain experts.

---

## 👨‍💻 Author

**Sushanth**  
Machine Learning | Data Analytics | AI Projects

🔗 GitHub: https://github.com/sushanth0963
  
