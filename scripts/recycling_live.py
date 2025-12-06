import time
from pathlib import Path

import cv2
import depthai as dai
from ultralytics import YOLO

# Path to your trained model (copied from Colab/notebook)
MODEL_PATH = Path("/home/kyle/models/warp_yolov8m_best.pt")


def create_pipeline():
    """
    Build a simple OAK-D pipeline that outputs 720p/1080p color frames.
    """
    pipeline = dai.Pipeline()

    cam = pipeline.createColorCamera()
    # Use main RGB camera
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)

    # Sensor + output resolution
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(1280, 720)  # smaller than full HD -> faster on Pi

    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # Autofocus / exposure / white balance
    cam.initialControl.setAutoFocusMode(
        dai.RawCameraControl.AutoFocusMode.CONTINUOUS_VIDEO
    )
    cam.initialControl.setAutoExposureEnable()
    cam.initialControl.setAutoWhiteBalanceMode(
        dai.RawCameraControl.AutoWhiteBalanceMode.AUTO
    )

    # Output stream
    xout = pipeline.createXLinkOut()
    xout.setStreamName("video")
    cam.video.link(xout.input)

    return pipeline


def main():
    # 1. Load YOLO model (CPU)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    class_names = model.names  # dict: {class_id: "name", ...}
    print("Classes:", class_names)

    # 2. Start OAK-D pipeline
    pipeline = create_pipeline()
    print("Connecting to OAK-D...")
    with dai.Device(pipeline) as device:
        print("Connected:", device.getMxId())
        q_video = device.getOutputQueue("video", maxSize=4, blocking=True)

        last_time = time.time()

        print("Running live recycling detection. Press 'q' to quit.")
        while True:
            in_frame = q_video.get()
            frame = in_frame.getCvFrame()  # numpy array, BGR

            # 3. Run YOLO inference on this frame (CPU)
            # imgsz can be lowered (e.g., 416 or 320) if it's too slow
            results = model.predict(
                frame,
                imgsz=640,
                conf=0.4,
                device="cpu",
                verbose=False,
            )

            r = results[0]
            annotated = frame.copy()
            has_recycling = False

            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    label_name = class_names.get(cls_id, str(cls_id))
                    label = f"{label_name} {conf:.2f}"

                    has_recycling = True

                    # Draw box + label
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

            # 4. Overlay status text
            status_text = "RECYCLING DETECTED" if has_recycling else "NO RECYCLING"
            status_color = (0, 255, 0) if has_recycling else (0, 0, 255)
            cv2.putText(
                annotated,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                status_color,
                2,
                cv2.LINE_AA,
            )

            # FPS overlay
            now = time.time()
            fps = 1.0 / (now - last_time)
            last_time = now
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # 5. Show live annotated feed
            cv2.imshow("Recycling Detector (press 'q' to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
