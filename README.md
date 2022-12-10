# Object Tracking with YOLOv7 and DeepSort

#### Language and Libraries

<p>
<a><img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen" alt="python"/></a>
<a><img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas"/></a>
<a><img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" alt="numpy"/></a>
<a><img src="https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white" alt="opencv"/></a>
<a><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="pytorch"/></a>
<a><img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)" alt="docker"/></a>
<a><img src="https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white" alt="gcp"/></a>
</p>

## Problem statement
Computer vision includes object tracking as a key component. 
Particularly in the areas of surveillance, traffic management, human-robot interaction, and imaging in medicine. 
Tracking is the process of determining an object's trajectory in following frames from its bounding box, or its position and size during the first frame.

## Solution Proposed

The solution proposed for the above problem to tracks the object movements in space or across different camera angles and to achieve the goal used Yolov7 with DeepSort.

## Model Used

The YOLO (You Only Look Once) v7 model is the latest in the family of YOLO models. 
YOLO models are single stage object detectors. In a YOLO model, image frames are featurized through a backbone. 
These features are combined and mixed in the neck, and then they are passed along to the head of the network YOLO predicts the locations and classes of objects around which bounding boxes should be drawn.

## How to run?

### Step 1: Clone the repository
```bash
git clone my repository 
```

### Step 2- Create a conda environment after opening the repository

```bash
conda create -p env python=3.7 -y
```

```bash
conda activate env
```

### Step 3 - Install the requirements
```bash
pip install -r requirements.txt
```

### Step 4 - Install Google Cloud Sdk and configure

#### For Windows
```bash
https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe
```
#### For Ubuntu
```bash
sudo apt-get install apt-transport-https ca-certificates gnupg
```
```bash
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
```
```bash
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
```
```bash
sudo apt-get update && sudo apt-get install google-cloud-cli
```
```bash
gcloud init
```
Before running server application make sure your `Google Cloud Storage` bucket is available

### Step 5 - Run the application server
```bash
python app.py
```

## Run locally

1. Check if the Dockerfile is available in the project directory

2. Build the Docker image

```
docker build -t track . 
```

3. Run the Docker image

```
docker run -d -p 8080:8080 <IMAGEID>
```

👨‍💻 Tech Stack Used
1. Python
2. Pytorch
3. Docker
4. Yolov7
5. Deep SORT

🌐 Infrastructure Required.
1. Google Cloud Storage
2. Google Compute Engine
3. Google Artifact Registry
4. Circle CI


## `src` is the main package folder which contains 

**Artifact** : Stores all artifacts created from running the application

**Components** : Contains all components of this project
- ModelIngestion
- ModelLoading
- DataTransformation
- ObjectTracking
- Pusher

**Custom Logger and Exceptions** are used in the project for better debugging purposes.


## Conclusion



=====================================================================