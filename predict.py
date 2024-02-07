from ultralytics import YOLO
import os
from PIL import Image

model = YOLO("./runs/segment/train/weights/best.pt")

# results = model.predict(source="test/images",save=True,device=1)  # predict on an image

results = model.predict("test/images/", save=True, device=1) # conf=0.3, save=True "../test/images"

# for res in results:
#     print("*********************")
#     # print(res)
#     try:
#         h,e = res.masks.data[0].size()
#     except:
#         h,e = res.orig_shape
#     print(h,e)
#     # print(res.boxes,res.masks)
#     if not (res.boxes == None or res.masks == None):
#         for c,m in zip(res.boxes.cls,res.masks.data):
#             print(res.names[c.cpu().item()])
#             print(m.size())
#             print(res.orig_shape)
#             out_np = m.cpu().numpy().astype(int)

# print(results[0].names[results[0][0].boxes.cls.item()])
# print(results[0][0].boxes.cls)
# print(results[0][0].masks.data.size())
# print(sum(results[0][0].masks.data[0]))

# print(results[0][0].masks.xyn)
# print(results[0][1].boxes.xywh)

# orig_h, orig_w = results[0][0].masks.orig_shape
# print(orig_w, orig_h)
# out = results[0][0].masks.data[0]
# out = out.cpu().numpy().astype(int)
# out[out > 1] = 1
# gray_img = Image.fromarray(out.astype('uint8')*255)
# o = gray_img.convert('L')
# o = o.resize(size=(orig_w, orig_h))
# o.save("test_predict.jpg")