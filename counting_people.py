import cv2
from ultralytics import YOLO
def counting_people(video_file=r"output/output.avi",conf_threshold= 0.5,frame_width = 640,frame_height = 480):
    model_path = 'yolov8n-seg.pt'
    yolo_model = YOLO(model_path)

    # Open video
    video = cv2.VideoCapture(video_file)
    assert video.isOpened(), "Cannot open video"
    
    while True:
        ret, frame = video.read()
        if not ret:
            print('Video processing completed')
            break
    
        frame = cv2.resize(frame, (frame_width, frame_height))
    
        results = yolo_model(frame)[0]
        person_count = 0
    
        boxes = results.boxes
        for box in boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
    
            if confidence > conf_threshold and yolo_model.names[class_id] == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
    
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
                label = f"{yolo_model.names[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
    
                person_count += 1
    
        cv2.putText(frame, f"Count: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
        cv2.imshow("Video", frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    video.release()
    cv2.destroyAllWindows()
    