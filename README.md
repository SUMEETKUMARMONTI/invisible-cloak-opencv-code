<div align="center">

# 🧥 Invisible Cloak using OpenCV

### Make yourself disappear with the power of Computer Vision!

**A real-time computer vision project that uses Python, OpenCV, and background replacement techniques to create an invisible cloak effect through your webcam.**

<br>

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge\&logo=opencv\&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=for-the-badge\&logo=numpy\&logoColor=white)](https://numpy.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey?style=for-the-badge)](#)
[![Status](https://img.shields.io/badge/Project-Completed-success?style=for-the-badge)](#)

<br>

**Developed by [SUMEET KUMAR MONTI](https://github.com/SUMEETKUMARMONTI)**

[GitHub Profile](https://github.com/SUMEETKUMARMONTI) • [LinkedIn](https://www.linkedin.com/in/sumeetkumarmonti932003/)

</div>

---

## ✨ Project Preview

<div align="center">

<img src="screenshots/output.png" alt="Invisible Cloak Project Preview" width="85%">

**The Invisible Cloak Effect — Powered by OpenCV**

</div>

> The preview image demonstrates the intended visual effect. For a technical demonstration, you can also add a genuine screenshot captured while running the application.

---

## 📌 About the Project

**Invisible Cloak using OpenCV** is a real-time computer vision application inspired by the invisible cloak concept popularized in science-fiction movies.

The project uses a webcam to capture the environment, identify a specific cloak color (green by default), and replace the detected region with a previously captured background frame.

By combining image processing, color detection, masking, and bitwise operations, the application creates the illusion that a person wearing the selected colored cloth has disappeared.

This project demonstrates how fundamental computer vision techniques can be used to build an interactive visual effect using Python.

### 🎯 Project Objectives

* Understand the fundamentals of computer vision.
* Implement real-time image processing using OpenCV.
* Detect a specific color using HSV color space.
* Apply image masking and morphological operations.
* Understand background capture and image compositing.
* Build an interactive webcam-based application.

---

## 🚀 Key Features

| Feature                         | Description                                                                    |
| ------------------------------- | ------------------------------------------------------------------------------ |
| 🎥 Real-Time Webcam             | Captures live video using your webcam.                                         |
| 🧥 Invisible Cloak Effect       | Replaces the detected cloak-colored region with the captured background.       |
| 🎨 HSV Color Detection          | Uses HSV thresholding to identify the selected cloak color.                    |
| 🎚️ Adjustable Color Thresholds | Provides trackbars to tune HSV detection values.                               |
| 🖼️ Background Capture          | Captures and stores the background frame for replacement.                      |
| 🔄 Background Recapture         | Allows you to capture the background again without restarting the application. |
| 🧹 Noise Reduction              | Uses morphological operations to improve mask quality.                         |
| 📊 FPS Display                  | Displays approximate real-time processing performance.                         |
| 📸 Screenshot Capture           | Saves the processed output with a timestamp.                                   |
| 👁️ Mask Preview                | Allows you to view the detected color mask.                                    |
| ⏸️ Pause / Resume               | Temporarily pauses and resumes the video feed.                                 |
| 🏷️ Developer Branding          | Displays developer information and social profile links in the application.    |

---

## 🛠️ Technology Stack

| Technology             | Purpose                                      |
| ---------------------- | -------------------------------------------- |
| **Python**             | Core programming language                    |
| **OpenCV**             | Image processing and computer vision         |
| **NumPy**              | Array manipulation and image operations      |
| **Webcam**             | Live video input                             |
| **HSV Color Space**    | Color segmentation and detection             |
| **Bitwise Operations** | Background replacement and image compositing |

---

## ⚙️ How Does It Work?

The application follows a sequence of computer vision operations to produce the invisible cloak effect.

### 1. Background Capture

The application captures the empty background before the person enters the camera frame.

### 2. Live Video Capture

The webcam continuously captures frames and sends them for processing.

### 3. Color Detection

The current frame is converted from BGR to HSV color space. The application identifies pixels within the selected color range.

### 4. Mask Generation

A binary mask is generated to separate the cloak-colored region from the remaining image.

### 5. Noise Reduction

Morphological operations such as:

* Erosion / Opening
* Closing
* Dilation

are applied to improve the mask and reduce unwanted noise.

### 6. Background Replacement

Bitwise operations combine the captured background and the live camera frame.

The detected cloak-colored region is replaced with the corresponding region from the background image.

### 7. Final Output

The processed frames are displayed continuously, creating the illusion of invisibility.

---

## 📂 Project Structure

```text
Invisible-Cloak-OpenCV/
│
├── docs/
│   ├── Invisible_Cloak_Project_Run_Guide.pdf
│   └── Invisible_Cloak_Project_Run_Guide.docx
│
├── output/
│   └──                 # Generated screenshots
│
├── screenshots/
│   └── output.png      # Project preview
│
├── Invisible_Cloak.py  # Main application
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
├── .gitignore          # Git ignored files
└── project_structure.txt
```

> The `venv/` and `__pycache__/` directories are local development/generated files and should not be uploaded to the repository.

---

## 💻 Installation & Setup

Follow these steps to run the project on your local machine.

### Prerequisites

Make sure you have:

* Python 3.x installed.
* A working webcam.
* Visual Studio Code or another Python editor.
* Git (optional, for cloning the repository).

### Step 1: Clone the Repository

```bash
git clone https://github.com/SUMEETKUMARMONTI/Invisible-Cloak-OpenCV.git
```

### Step 2: Navigate to the Project Folder

```bash
cd Invisible-Cloak-OpenCV
```

### Step 3: Create a Virtual Environment

**Windows:**

```powershell
py -m venv venv
```

### Step 4: Install Dependencies

**Windows:**

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Step 5: Run the Application

```powershell
.\venv\Scripts\python.exe Invisible_Cloak.py
```

**That's it! Your Invisible Cloak application should now start using your webcam.**

> Make sure your operating system allows webcam access and that no other application is exclusively using the camera.

---

## 📦 Dependencies

The project uses the following Python libraries:

```text
opencv-python
numpy
```

They are listed in `requirements.txt` and can be installed using:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## ⌨️ Keyboard Shortcuts

The application supports the following keyboard controls:

| Key     | Action                             |
| ------- | ---------------------------------- |
| `Q`     | Exit the application               |
| `ESC`   | Close the application              |
| `B`     | Capture the background again       |
| `S`     | Save the current output screenshot |
| `M`     | Toggle mask preview                |
| `H`     | Toggle information panel           |
| `SPACE` | Pause / Resume video               |

---

## 📸 Screenshot Output

Screenshots captured through the application are saved in the `output/` directory.

Each screenshot uses a timestamp-based filename to help keep multiple captures organized.

Example:

```text
output/
├── invisible_cloak_2026-11-03_120530.png
└── invisible_cloak_2026-11-03_121045.png
```

---

## 🧠 Concepts Demonstrated

This project helped me explore and implement several important computer vision concepts:

* Image acquisition using OpenCV.
* BGR and HSV color spaces.
* Color segmentation and thresholding.
* Binary image masks.
* Morphological image processing.
* Image filtering and noise reduction.
* Bitwise image operations.
* Background subtraction and replacement.
* Real-time video processing.
* Webcam handling and resource management.

---

## ⚠️ Limitations

Although the project creates an interesting visual effect, it has some practical limitations:

1. **Lighting Conditions:** Changes in lighting can affect color detection.
2. **Background Stability:** The background should remain relatively unchanged after capture.
3. **Cloak Color:** The selected color should be distinct from the surrounding environment.
4. **Camera Position:** The camera should remain stationary for better results.
5. **Color Interference:** Objects with similar colors may also be detected as part of the cloak.
6. **Webcam Quality:** Lower-quality cameras may affect the final output.

For the best results, use a solid green cloth, stable lighting, and a fixed camera position.

---

## 🔮 Future Improvements

Some potential improvements for future versions include:

* Support for multiple cloak colors.
* Automatic color calibration.
* Improved segmentation using advanced computer vision techniques.
* Better handling of changing lighting conditions.
* A graphical user interface for configuration.
* Video recording functionality.
* More advanced background replacement techniques.
* Performance optimization for lower-end systems.

---

## 📚 Learning Resources & Acknowledgement

This project was inspired by the **Invisible Cloak using OpenCV** tutorial published by DataFlair.

I would like to acknowledge the original tutorial and learning resources that helped me understand the core concept and implementation.

**This repository is my learning and development implementation of the concept, with additional application features and documentation.**

---

## 👨‍💻 About the Developer

<div align="center">

### SUMEET KUMAR MONTI

**Computer Science & Engineering Graduate | Software Development | AI & Computer Vision Enthusiast**

I enjoy exploring technologies, building practical applications, and learning how software can solve real-world problems.

This project represents my interest in Python programming, computer vision, and hands-on software development.

<br>

[![GitHub](https://img.shields.io/badge/GitHub-SUMEETKUMARMONTI-181717?style=for-the-badge\&logo=github)](https://github.com/SUMEETKUMARMONTI)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect%20with%20Me-0A66C2?style=for-the-badge\&logo=linkedin)](https://www.linkedin.com/in/sumeetkumarmonti932003/)

</div>

---

## ⭐ Support

If you find this project interesting or useful:

* Give this repository a ⭐ on GitHub.
* Explore the source code.
* Share your feedback and suggestions.
* Connect with me on LinkedIn.

Your feedback and suggestions are always welcome!

---

<div align="center">

### Built with ❤️, Python & OpenCV

**© 2026 Sumeet Kumar Monti.**

*Keep Learning. Keep Building. Keep Growing.*

</div>
