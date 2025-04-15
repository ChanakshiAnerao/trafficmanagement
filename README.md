# 🚦 Smart Traffic Management System using YOLOv8, OpenCV & Pygame

An AI-powered **Vehicle Detection, Counting, and Dynamic Traffic Signal Control System** using YOLOv8 for object detection and Pygame/Matplotlib for real-time graphical traffic signal simulation.

This project mimics a real-world smart traffic control system where the duration of traffic lights dynamically adjusts based on the number of vehicles detected in each lane.

---

## 📌 Project Summary

This project uses a camera (or video feed) to:
- Detect and count vehicles in real-time using **YOLOv8**
- Dynamically adjust traffic light durations based on vehicle density
- Visually simulate traffic light signals using **Pygame** or **Matplotlib**

---

## 🧠 Technologies Used

| Component              | Description |
|------------------------|-------------|
| 🔍 YOLOv8 (Ultralytics) | Real-time, accurate vehicle detection |
| 🎥 OpenCV              | Frame capture and image processing |
| 🔢 NumPy               | Numerical operations and array handling |
| 🎮 Pygame / 📊 Matplotlib | Visual representation of traffic signals |
| 🧮 Custom Logic         | Lane-wise vehicle counting and adaptive signal timing |

---

## 📦 Requirements

Install all dependencies using pip:

```bash
pip install ultralytics opencv-python numpy pygame matplotlib
Or from requirements.txt:

bash
Copy
Edit
pip install -r requirements.txt# trafficmanagement
