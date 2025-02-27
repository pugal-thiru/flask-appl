from flask import Flask, Response, render_template
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
from PIL import Image
import numpy as np
import atexit

app = Flask(__name__)

# Define the number of classes
num_classes = 2  # Example: 1 object class + background

# Load the Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Load trained weights
state_dict = torch.load("mobile_detection_model.pth", map_location=torch.device("cpu"))
model.load_state_dict(state_dict, strict=True)  # Ensuring strict loading
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define image transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

# Open webcam
cap = cv2.VideoCapture(0)

def release_camera():
    """ Release webcam when the app stops """
    cap.release()
    cv2.destroyAllWindows()

atexit.register(release_camera)  # Ensure webcam is closed when exiting

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL Image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(img_tensor)

        for pred in predictions:
            for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
                if score > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"Class {label}: {score:.2f}"
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
