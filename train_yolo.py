from ultralytics import YOLO
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 
# import torch
# torch.cuda.set_device(1)

# Load a model
model = YOLO('yolov8l-seg.pt')  # load a pretrained model (recommended for training)
results = model.train(data="data.yaml", epochs=100,  device=1)
# Validate the model
results = model.val(data="data.yaml")
