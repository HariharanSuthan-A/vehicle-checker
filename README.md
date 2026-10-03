# 🚔 Vehicle Checker — License Plate Detector

An AI-powered Streamlit app that detects and logs license plates from **bikes/motorcycles** and **cars/trucks** using custom YOLOv8 models.

---

## 🚀 Features

- 🤖 **Two YOLO models** — switch between bike and car/truck detection
- 🖼️ **Sample images** — pre-loaded test images for each vehicle type
- 📤 **Upload your own image** — JPG, PNG, WEBP supported
- 🪪 **Plate cropping** — automatically crops each detected plate
- 📋 **Detection log table** — stores every result with date, time, confidence & plate thumbnail
- 📦 **Export** — download all cropped plates as ZIP or the log as CSV
- ⚙️ **Adjustable thresholds** — confidence & IoU sliders in the sidebar

---

## 📁 Project Structure

```
vehicle-checker/
├── app.py              # Streamlit application
├── bikemodel.pt        # YOLOv8 model — bike license plates
├── carmodel.pt         # YOLOv8 model — car/truck license plates
├── requirements.txt    # Python dependencies
└── samples/            # Sample test images
    ├── bike.jpeg
    ├── bike3.jpg
    ├── bike5.jpg
    ├── bike7.jpg
    ├── car3.jpg
    ├── car5.jpg
    └── car6.jpg
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/vehicle-checker.git
cd vehicle-checker

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| UI | Streamlit |
| Detection | Ultralytics YOLOv8 |
| Image processing | Pillow, OpenCV |
| Data | Pandas, NumPy |

---

## 📝 License

MIT
