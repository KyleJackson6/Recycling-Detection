import time
import cv2
import numpy as np
import depthai as dai
import onnxruntime as ort


# -----------------------------
# Config
# -----------------------------
# ### CHANGED: point this to your NEW nano ONNX file
MODEL_PATH = "/home/kyle/Recycling-Detection/models/warp_yolov8n_416.onnx"

# ### CHANGED: model input size (you trained/exported at 416x416)
MODEL_W = 416
MODEL_H = 416

CONF_THRESH = 0.35
IOU_THRESH = 0.5

# Run ONNX every N frames to reduce CPU load
INFER_EVERY_N = 2  # you can bump this to 3 if needed

# Class names (from your WaRP dataset)
CLASS_NAMES = [
    "bottle-blue",        # 0
    "bottle-green",       # 1
    "bottle-dark",        # 2
    "bottle-milk",        # 3
    "bottle-transp",      # 4
    "bottle-multicolor",  # 5
    "bottle-yogurt",      # 6
    "bottle-oil",         # 7
    "cans",               # 8
    "juice-cardboard",    # 9
    "milk-cardboard",     # 10
    "detergent-color",    # 11
    "detergent-transparent",  # 12
    "detergent-box",      # 13
    "canister",           # 14
    "bottle-blue-full",   # 15
    "bottle-transp-full", # 16
    "bottle-dark-full",   # 17
    "bottle-green-full",  # 18
    "bottle-multicolorv-full",  # 19
    "bottle-milk-full",   # 20
    "bottle-oil-full",    # 21
    "detergent-white",    # 22
    "bottle-bluesL",      # 23
    "bottle-bluesL-full", # 24
    "glass-transp",       # 25
    "glass-dark",         # 26
    "glass-green",        # 27
]


# -----------------------------
# DepthAI pipeline (same style as your working live_view.py)
# -----------------------------
def create_pipeline(preview_w, preview_h):
    pipeline = dai.Pipeline()

    cam = pipeline.createColorCamera()
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)

    # Use 1080p sensor mode, but get a square preview for detection
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setPreviewSize(preview_w, preview_h)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # SAME autofocus / exposure logic as your working live_view.py
    cam.initialControl.setAutoFocusMode(dai.RawCameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
    cam.initialControl.setAutoExposureEnable()
    cam.initialControl.setAutoWhiteBalanceMode(
        dai.RawCameraControl.AutoWhiteBalanceMode.AUTO
    )

    # XLink output for preview frames
    xout = pipeline.createXLinkOut()
    xout.setStreamName("preview")
    cam.preview.link(xout.input)

    return pipeline


# -----------------------------
# YOLO helper functions
# -----------------------------
def preprocess(frame, input_w, input_h):
    """
    Resize BGR frame to (input_w, input_h), convert to NCHW float32 /255.
    """
    img = cv2.resize(frame, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, 0)        # -> NCHW
    return img


def xywh2xyxy(x):
    # x: (N,4) with [cx, cy, w, h]
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y


def nms(boxes, scores, iou_threshold):
    """
    Basic NMS in numpy.
    boxes: (N,4) xyxy
    scores: (N,)
    returns indices to keep
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def run_yolo(session, input_name, img_input, conf_thresh, iou_thresh):
    """
    Run YOLOv8 ONNX and return list of detections:
    each det = (x1, y1, x2, y2, score, cls_id)
    """
    outputs = session.run(None, {input_name: img_input})
    out = outputs[0]  # could be [1, N, 4+nc] OR [1, 4+nc, N]

    # ### CHANGED: robust handling for Ultralytics ONNX layout
    # out.shape is typically (1, 4+nc, N_anchors) for YOLOv8 ONNX
    out = out[0]  # drop batch -> (C, N) or (N, C)

    if out.shape[0] < out.shape[1]:
        # shape is (C, N) = (4+nc, N), transpose to (N, 4+nc)
        preds = out.transpose(1, 0)
    else:
        # shape already (N, 4+nc)
        preds = out

    boxes = preds[:, :4]
    scores_all = preds[:, 4:]

    # Best class per anchor
    cls_ids = np.argmax(scores_all, axis=1)
    scores = scores_all[np.arange(scores_all.shape[0]), cls_ids]

    # Filter by confidence
    mask = scores >= conf_thresh
    boxes = boxes[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]

    if len(boxes) == 0:
        return []

    # YOLO outputs [cx,cy,w,h], convert to [x1,y1,x2,y2]
    boxes_xyxy = xywh2xyxy(boxes)

    keep = nms(boxes_xyxy, scores, iou_thresh)
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    cls_ids = cls_ids[keep]

    detections = []
    for box, score, cid in zip(boxes_xyxy, scores, cls_ids):
        detections.append((box[0], box[1], box[2], box[3], float(score), int(cid)))

    return detections


# -----------------------------
# Main
# -----------------------------
def main():
    print(f"Loading ONNX model: {MODEL_PATH}")
    sess_options = ort.SessionOptions()
    # You can tweak graph optimization if needed
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],  # <-- don't touch your env/GPU
    )

    input_name = session.get_inputs()[0].name
    in_shape = session.get_inputs()[0].shape  # [1, 3, H, W]
    print("Model input shape from ONNX:", in_shape)
    print(f"Using input HxW: {MODEL_H}x{MODEL_W}")

    # Create DepthAI pipeline
    # NOTE: preview size is still 640x640 from the camera, but we resize to 416 in preprocess
    pipeline = create_pipeline(640, 640)

    print("Connecting to OAK-D...")
    with dai.Device(pipeline) as device:
        print("Connected:", device.getMxId())
        q_preview = device.getOutputQueue("preview", maxSize=4, blocking=True)

        last_detections = []
        frame_count = 0
        last_fps_time = time.time()
        fps = 0.0

        while True:
            in_frame = q_preview.get()
            frame = in_frame.getCvFrame()  # e.g. 640x640

            # Only run ONNX every Nth frame
            run_infer = (frame_count % INFER_EVERY_N == 0)

            if run_infer:
                img_input = preprocess(frame, MODEL_W, MODEL_H)
                dets = run_yolo(session, input_name, img_input, CONF_THRESH, IOU_THRESH)
                last_detections = dets
            else:
                dets = last_detections

            # Draw detections
            h, w, _ = frame.shape
            for (x1, y1, x2, y2, score, cid) in dets:
                # These coords are in model space (416x416), but because we resize
                # the 640x640 frame down to 416 and back, scales are consistent.
                x1 = int(max(0, min(w - 1, x1)))
                y1 = int(max(0, min(h - 1, y1)))
                x2 = int(max(0, min(w - 1, x2)))
                y2 = int(max(0, min(h - 1, y2)))

                label = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else f"id{cid}"
                label_text = f"{label} {score:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label_text,
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            # Status + FPS text
            status_text = "RECYCLING DETECTED" if len(dets) > 0 else "NO RECYCLING"
            status_color = (0, 255, 0) if len(dets) > 0 else (0, 0, 255)

            # Update FPS once per second (loop FPS, not inference FPS)
            now = time.time()
            dt = now - last_fps_time
            if dt > 1.0:
                fps = frame_count / dt
                frame_count = 0
                last_fps_time = now
            frame_count += 1

            cv2.putText(
                frame,
                f"{status_text}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Loop FPS: {fps:.1f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Recycling Detection (ONNX + OAK-D)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
