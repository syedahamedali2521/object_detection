# YOLO Object Detection App

A simple **Streamlit app** for object detection using **YOLOv8** and **OpenCV**.

The app can detect objects in images and display the results in real-time.

---

## 🔗 Live Demo

You can try the app online using the ngrok link:  

[https://uninveigled-misformed-maribel.ngrok-free.dev](https://uninveigled-misformed-maribel.ngrok-free.dev)

> Note: This link is temporary and will stop working when the local server is closed.

---

## 📂 Repository Structure

object_detection_app_v1/
├── app/
│ ├── app.py # Streamlit application
│ └── static/ # CSS, images, and other static files
├── requirements.txt # Python dependencies
└── README.md # This file

yaml
Copy code

---

## ⚡ Installation (Local Deployment)

1. Clone the repository:

```bash
git clone https://github.com/syedahamedali2521/object_detection_app_v1.git
cd object_detection_app_v1
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run .\app\app.py
(Optional) Expose your app publicly with ngrok:

bash
Copy code
ngrok http 8501
🛠 Dependencies
streamlit

numpy

Pillow

opencv-python-headless

ultralytics

All dependencies are listed in requirements.txt.

🖥 Usage
Upload an image via the app interface.

The app will detect objects and show bounding boxes and labels.

You can download the processed image with detected objects.

📄 License
This project is licensed under the MIT License.

💡 Notes
