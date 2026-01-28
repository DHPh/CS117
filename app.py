import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import timm

# Cấu hình trang
st.set_page_config(
    page_title="Sugarcane Disease Prediction",
    layout="wide"
)

# Cấu hình model
MODEL_NAME = "timm/convnextv2_tiny.fcmae"
MODEL_PATH = "best_model.pt"
INPUT_SIZE = 224
CLASS_NAMES = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]
NUM_CLASSES = len(CLASS_NAMES)

@st.cache_resource
def load_model():
    """Load model và cache để không phải load lại nhiều lần"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tạo model architecture
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES
    )
    
    # Thay thế classifier layer với dropout
    original_fc = model.head.fc
    model.head.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(original_fc.in_features, NUM_CLASSES)
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

def get_transform():
    """Transform cho inference"""
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def validate_image_constraints(image):
    """
    Kiểm tra ảnh có thỏa mãn các constraints hay không
    
    Returns:
        is_valid (bool): True nếu thỏa mãn tất cả constraints
        message (str): Thông báo chi tiết
        details (dict): Chi tiết các kiểm tra
    """
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    
    # Convert RGB to BGR cho OpenCV
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        return False, "Ảnh phải có 3 kênh màu (RGB)", {}
    
    height, width = image_cv.shape[:2]
    details = {}
    all_constraints_passed = True
    constraint_messages = []
    
    # Constraint 1: Resolution Validation
    MIN_WIDTH = 1280
    MIN_HEIGHT = 720
    resolution_valid = (width >= MIN_WIDTH and height >= MIN_HEIGHT)
    details['resolution'] = {
        'width': width,
        'height': height,
        'valid': resolution_valid,
        'required': f'{MIN_WIDTH}x{MIN_HEIGHT}'
    }
    
    if not resolution_valid:
        all_constraints_passed = False
        constraint_messages.append(f"Độ phân giải quá thấp: {width}x{height} (yêu cầu >= {MIN_WIDTH}x{MIN_HEIGHT})")
    
    # Constraint 2: Lighting Analysis
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    average_brightness = cv2.mean(gray)[0]
    
    THRESHOLD_MIN = 50
    THRESHOLD_MAX = 200
    lighting_valid = (THRESHOLD_MIN < average_brightness < THRESHOLD_MAX)
    details['lighting'] = {
        'brightness': average_brightness,
        'valid': lighting_valid,
        'range': f'{THRESHOLD_MIN}-{THRESHOLD_MAX}'
    }
    
    if not lighting_valid:
        all_constraints_passed = False
        constraint_messages.append(f"Độ sáng không phù hợp: {average_brightness:.1f} (yêu cầu {THRESHOLD_MIN}-{THRESHOLD_MAX})")
    
    # Constraint 3 & 4: Leaf Detection và Coverage
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    
    # Định nghĩa green color range cho lá mía
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Tạo binary mask
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Coverage Ratio
    leaf_area = cv2.countNonZero(mask)
    total_area = width * height
    coverage_ratio = leaf_area / total_area
    
    MIN_COVERAGE = 0.40
    coverage_valid = (coverage_ratio >= MIN_COVERAGE)
    details['coverage'] = {
        'leaf_area': leaf_area,
        'total_area': total_area,
        'ratio': coverage_ratio,
        'percentage': coverage_ratio * 100,
        'valid': coverage_valid,
        'required': f'>={MIN_COVERAGE*100}%'
    }
    
    if not coverage_valid:
        all_constraints_passed = False
        constraint_messages.append(f"Diện tích lá quá ít: {coverage_ratio*100:.1f}% (yêu cầu >= {MIN_COVERAGE*100}%)")
    
    # Constraint 5: Centering Check
    M = cv2.moments(mask)
    
    if M["m00"] == 0:
        all_constraints_passed = False
        constraint_messages.append("Không phát hiện được lá trong ảnh")
        details['centering'] = {'valid': False}
    else:
        # Tính tọa độ trung tâm của lá
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        # Định nghĩa vùng center box
        center_margin = 0.3
        x_min = int(width * center_margin)
        x_max = int(width * (1 - center_margin))
        y_min = int(height * center_margin)
        y_max = int(height * (1 - center_margin))
        
        centering_valid = (x_min <= cX <= x_max and y_min <= cY <= y_max)
        details['centering'] = {
            'center_x': cX,
            'center_y': cY,
            'valid': centering_valid,
            'required_region': f'({x_min}-{x_max}, {y_min}-{y_max})'
        }
        
        if not centering_valid:
            all_constraints_passed = False
            constraint_messages.append(f"Lá không ở giữa khung hình: vị trí ({cX}, {cY})")
    
    if all_constraints_passed:
        return True, "Ảnh hợp lệ, đáp ứng tất cả các yêu cầu", details
    else:
        return False, " | ".join(constraint_messages), details

def predict_image(model, image, transform, device):
    """
    Dự đoán class và confidence cho một ảnh
    
    Returns:
        predicted_class (int): Index của class dự đoán
        confidence (float): Confidence score (0-1)
        all_probs (tensor): Probability của tất cả classes
    """
    # Preprocess image
    img_tensor = transform(image).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(img_tensor)
    
    # Apply Softmax
    probabilities = F.softmax(logits, dim=1)
    
    # Get predicted class và confidence
    confidence, predicted_class = probabilities.max(1)
    
    return predicted_class.item(), confidence.item(), probabilities[0]

# Main App
def main():
    st.title("Dự đoán bệnh lá mía")
    st.write("Upload ảnh lá mía để dự đoán bệnh")
    
    # Load model
    try:
        model, device = load_model()
        transform = get_transform()
        st.success(f"Model đã được tải thành công (Device: {device})")
    except Exception as e:
        st.error(f"Lỗi khi tải model: {str(e)}")
        return
    
    # Tùy chọn bỏ qua kiểm tra
    skip_validation = st.checkbox("Bỏ qua kiểm tra ràng buộc (dự đoán ngay cả khi ảnh không hợp lệ)")
    
    # Upload file
    uploaded_file = st.file_uploader("Chọn ảnh lá mía", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Đọc ảnh
        image = Image.open(uploaded_file).convert('RGB')
        
        # Chia làm 2 cột
        col1, col2 = st.columns([1, 1])
        
        # Cột 1: Hiển thị ảnh
        with col1:
            st.subheader("Ảnh đầu vào")
            st.image(image, caption="Ảnh đã upload", use_container_width=True)
        
        # Cột 2: Kiểm tra và dự đoán
        with col2:
            # Kiểm tra constraints
            st.subheader("Kiểm tra tính hợp lệ")
            
            with st.spinner("Đang kiểm tra ảnh..."):
                is_valid, message, details = validate_image_constraints(image)
            
            # Hiển thị trạng thái kiểm tra
            if is_valid:
                st.success(message)
            else:
                st.error(f"Ảnh không hợp lệ: {message}")
            
            # Chi tiết kiểm tra (luôn hiển thị)
            with st.expander("Xem chi tiết kiểm tra"):
                if 'resolution' in details:
                    res = details['resolution']
                    status = "Đạt" if res['valid'] else "Không đạt"
                    st.write(f"**Độ phân giải:** {status} - {res['width']}x{res['height']} (Yêu cầu: >={res['required']})")
                
                if 'lighting' in details:
                    light = details['lighting']
                    status = "Đạt" if light['valid'] else "Không đạt"
                    st.write(f"**Độ sáng:** {status} - {light['brightness']:.2f} (Yêu cầu: {light['range']})")
                
                if 'coverage' in details:
                    cov = details['coverage']
                    status = "Đạt" if cov['valid'] else "Không đạt"
                    st.write(f"**Diện tích lá:** {status} - {cov['percentage']:.2f}% (Yêu cầu: {cov['required']})")
                
                if 'centering' in details:
                    cent = details['centering']
                    if cent['valid']:
                        st.write(f"**Vị trí lá:** Đạt - ({cent['center_x']}, {cent['center_y']})")
                    else:
                        if 'center_x' in cent:
                            st.write(f"**Vị trí lá:** Không đạt - ({cent['center_x']}, {cent['center_y']})")
                        else:
                            st.write(f"**Vị trí lá:** Không đạt - Không phát hiện được lá")
            
            # Dự đoán nếu ảnh hợp lệ HOẶC người dùng chọn bỏ qua kiểm tra
            if is_valid or skip_validation:
                st.subheader("Kết quả dự đoán")
                
                if not is_valid:
                    st.info("Đang dự đoán mặc dù ảnh không đạt yêu cầu")
                
                with st.spinner("Đang dự đoán..."):
                    pred_class_idx, confidence, all_probs = predict_image(model, image, transform, device)
                    pred_class_name = CLASS_NAMES[pred_class_idx]
                
                # Format output
                if pred_class_name == "Healthy":
                    output_text = "Healthy - None"
                else:
                    output_text = f"Disease - {pred_class_name}"
                
                # Hiển thị kết quả với font lớn
                st.markdown(f"<h2 style='text-align: left; color: #1f77b4;'>{output_text}</h2>", unsafe_allow_html=True)
            
            else:
                st.warning("Vui lòng upload ảnh khác đáp ứng các yêu cầu hoặc chọn 'Bỏ qua kiểm tra ràng buộc'")

if __name__ == "__main__":
    main()
