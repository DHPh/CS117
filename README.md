# Sugarcane Disease Prediction

Ứng dụng dự đoán bệnh lá mía sử dụng Deep Learning với ConvNeXt V2.

## Cài đặt

### 1. Clone hoặc tải project

```bash
cd CS117-report
```

### 2. Tạo môi trường ảo (khuyến nghị)

**Với Conda:**

```bash
conda create -n sugarcane python=3.10
conda activate sugarcane
```

**Với venv:**

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

### Khởi động Streamlit app

```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở trong trình duyệt tại `http://localhost:8501`

### Sử dụng app

1. **Upload ảnh**: Click "Browse files" và chọn ảnh lá mía
2. **Kiểm tra tự động**: Ứng dụng sẽ kiểm tra các ràng buộc:
   - Độ phân giải: ≥ 1280x720
   - Độ sáng: 50-200
   - Diện tích lá: ≥ 40%
   - Vị trí lá: ở giữa khung hình
3. **Bỏ qua kiểm tra** (tùy chọn): Tích vào checkbox nếu muốn dự đoán mà không cần đáp ứng ràng buộc
4. **Xem kết quả**: Kết quả hiển thị dạng "Healthy - None" hoặc "Disease - [tên bệnh]"
