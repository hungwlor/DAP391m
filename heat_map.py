import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO
from skimage.transform import resize
def get_row_col(x, y):
    row = y // cell_size
    col = x // cell_size
    return row, col
    
def draw_grid(image):
    for i in range(n_rows):
        start_point = (0, (i+1) * cell_size)
        end_point = (frame_width, (i+1) * cell_size)
        image = cv2.line(image, start_point, end_point, (255, 255, 255), 1) 
        
    for i in range(n_cols):
        start_point = ((i+1) * cell_size, 0)
        end_point = ((i+1) * cell_size, frame_height)
        image = cv2.line(image, start_point, end_point, (255, 255, 255), 1) 
    return image

def update_heatmap_from_mask(mask, heat_matrix, n_rows, n_cols):
    resized_mask = resize(mask, (n_rows, n_cols), anti_aliasing=True)
    heat_matrix += resized_mask
    
def heat_map(video_file=r"output/output.avi",conf_threshold= 0.5,alpha=0.4,cell_size = 40):
    model_path = 'yolov8n-seg.pt'
    yolo_model = YOLO(model_path)

    # Mở video
    video = cv2.VideoCapture(video_file)
    assert video.isOpened(), "Không thể mở video"
    
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_cols = frame_width // cell_size
    n_rows = frame_height // cell_size
    heat_matrix = np.zeros((n_rows, n_cols))
    while True:
        ret, frame = video.read()
        if not ret:
            print('Hoàn tất xử lý video')
            break
    
        # Dự đoán với YOLOv8 segmentation
        results = yolo_model(frame)[0]
    
        if results.masks is not None:  # Kiểm tra nếu có masks
            for i, result in enumerate(results.boxes):
                class_id = int(result.cls[0])
                confidence = result.conf[0]
    
                # Kiểm tra nếu confidence đủ lớn và đối tượng là "person"
                if confidence > conf_threshold and yolo_model.names[class_id] == "person":
                    # Lấy mask tương ứng
                    mask = results.masks.data[i].cpu().numpy()  # Lấy mask tương ứng với bounding box hiện tại
    
                    # Cập nhật heatmap từ mask
                    update_heatmap_from_mask(mask, heat_matrix,n_rows, n_cols)
    
                    # Vẽ mask lên frame
                    mask_color = np.zeros_like(frame)
                    mask_color[:, :, 1] = (mask * 255).astype(np.uint8)  # Tô màu xanh (G) cho đối tượng
                    frame = cv2.addWeighted(frame, 1, mask_color, 0.6, 0)
    
        # Resize heatmap để khớp với kích thước frame
        temp_heat_matrix = heat_matrix.copy()
        temp_heat_matrix = resize(temp_heat_matrix, (frame_height, frame_width), anti_aliasing=True)
        temp_heat_matrix = (temp_heat_matrix / np.max(temp_heat_matrix) * 255).astype(np.uint8)
    
        # Chuyển đổi heatmap thành 3 kênh (BGR) nếu cần
        if len(temp_heat_matrix.shape) == 2:
            temp_heat_matrix = cv2.cvtColor(temp_heat_matrix, cv2.COLOR_GRAY2BGR)
    
        # Áp dụng colormap
        image_heat = cv2.applyColorMap(temp_heat_matrix, cv2.COLORMAP_JET)
    
        # Đảm bảo kích thước của heatmap khớp với frame
        image_heat = cv2.resize(image_heat, (frame_width, frame_height))
    
        # Kết hợp heatmap với frame
        combined_frame = cv2.addWeighted(image_heat, alpha, frame, 1 - alpha, 0)
    
        # Hiển thị video
        cv2.imshow("Video", combined_frame)
    
        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Giải phóng tài nguyên
    video.release()
    cv2.destroyAllWindows()




