# CHOCOLATE-DETECTION
This project demonstrates a real-time product and object detection system using a custom-trained YOLO model. The system captures video from the webcam, detects objects, and applies a smart blur effect to highlight detected items while keeping the rest of the frame blurred.

1) Data Collection & Labeling (Roboflow)

I captured my own dataset (various angles, lighting, backgrounds).

I uploaded the images to Roboflow and manually labeled each object/class.

I applied Roboflow preprocessing/augmentation (resize to 640, optional flips/rotations, blur, brightness/contrast, etc.).

I generated Train/Val/Test splits and then Exported the dataset in YOLOv5 format.

Result: a versioned dataset in Roboflow ready to be pulled from Colab.

2) Training on Google Colab (YOLOv5)

I used a Colab notebook with a free GPU (T4 when available).

I cloned the official YOLOv5 repo, installed deps, then pulled my Roboflow dataset.

# Clone YOLOv5
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt


Option A — Download via Roboflow SDK

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")     # put your API key here
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
dataset = project.version(YOUR_VERSION).download("yolov5")  # e.g., 1
# This creates a data.yaml at: /content/yolov5/{dataset.name}/data.yaml


Option B — Download the YOLOv5 ZIP from Roboflow

From Roboflow “Export” page, copy the “Download code” cell or download the ZIP and unzip into /content/yolov5/.

3) Start Training

I trained starting from a lightweight base (e.g., yolov5s.pt).

# Example training run
!python train.py \
  --img 640 \
  --batch 16 \
  --epochs 100 \
  --data /content/yolov5/<your_dataset_folder>/data.yaml \
  --weights yolov5s.pt \
  --name chief_yolov5


Notes

--data points to the data.yaml that Roboflow created (includes class names & paths).

Tune --epochs, --batch, and choose a larger base (e.g., yolov5m.pt) if you need more accuracy and have GPU memory.

Roboflow augmentations help generalization; you can also enable YOLOv5’s built-in aug by leaving defaults.

4) Best Weights Output

After training, YOLOv5 writes the best checkpoint to:

/content/yolov5/runs/train/chief_yolov5/weights/best.pt

5) Download best.pt to Local
from google.colab import files
files.download('/content/yolov5/runs/train/chief_yolov5/weights/best.pt')


I then placed best.pt in the project root (same folder as main.py).
That’s the exact file the app loads here:

# main.py
model = YOLO("best.pt")

6) Sanity Check & Inference

I verified predictions on a few sample frames/images before committing to the repo:

# (Optional) Test directly with YOLOv5 script on a sample image/folder
!python detect.py --weights runs/train/chief_yolov5/weights/best.pt --img 640 --source /path/to/images

7) Tips That Helped

Balanced splits: Ensure train/val/test reflect real usage (lighting, backgrounds).

Class names: Keep them consistent in Roboflow; they flow into data.yaml and then into labels displayed by the app.

Thresholds: In main.py I used conf=0.4; tune this based on your precision/recall needs.

Small boxes filter: I ignore tiny detections to reduce noise.
