import cv2
import depthai as dai

pipeline = dai.Pipeline()
cam = pipeline.createColorCamera()
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.video.link(xout.input)

with dai.Device(pipeline) as device:
    q = device.getOutputQueue("video", maxSize=4, blocking=False)
    while True:
        frame = q.get().getCvFrame()
        cv2.imshow("OAK TEST", frame)
        if cv2.waitKey(1) == ord('q'):
            break
