# Báo Cáo Dự Án Cá Nhân

## Tổng Quan

Dự án cá nhân này bao gồm nhiều module khác nhau, tập trung vào việc phát triển các ứng dụng và mô hình học máy liên quan đến dữ liệu, API, và trí tuệ nhân tạo. Dự án được tổ chức thành các thư mục riêng biệt cho từng chủ đề, bao gồm tin tức, YouTube, du lịch, AI, và phân tích dữ liệu.

## Các Module Chính

### 1. Module Tin Tức (1_news)
- **Mô tả**: Tích hợp API tin tức với backend Python (Flask) và frontend (có thể là React).
- **Tính năng**:
  - Backend: Ứng dụng Flask để xử lý API.
  - Frontend: Giao diện người dùng (có thể là ứng dụng React).
  - Test: Sử dụng NewsAPI để lấy dữ liệu tin tức.
- **Công nghệ**: Python, Flask, React (dự đoán), NewsAPI.

### 2. Module YouTube (2_youtube)
- **Mô tả**: Tích hợp với YouTube API để lấy và xử lý dữ liệu video.
- **Tính năng**:
  - Script demo đầy đủ cho YouTube API.
  - Test các chức năng API.
- **Công nghệ**: Python, YouTube Data API.

### 3. Module Du Lịch (3_travel)
- **Mô tả**: Ứng dụng du lịch sử dụng Google Maps Places API để tìm kiếm địa điểm, khách sạn, nhà hàng.
- **Tính năng**:
  - Lấy dữ liệu địa điểm từ RapidAPI (Google Maps Places).
  - Lưu trữ trong MongoDB Atlas.
  - Có nhiều phiên bản (v1, v2) với cải tiến.
  - Tạo tour du lịch dựa trên dữ liệu.
  - Xử lý dữ liệu địa điểm với file CSV worldcities.
- **Công nghệ**: Python, MongoDB, Google Maps API, RapidAPI.

### 4. Module Gemini (4_gemini)
- **Mô tả**: Test tích hợp với Gemini API (có thể là Google Gemini AI).
- **Tính năng**: Script test để kiểm tra chức năng API.
- **Công nghệ**: Python, Gemini API.

### 5. Module Kibana (5_kibana)
- **Mô tả**: Thiết lập Elasticsearch và Kibana bằng Docker.
- **Tính năng**: Docker Compose để chạy stack ELK (Elasticsearch, Logstash, Kibana).
- **Công nghệ**: Docker, Elasticsearch, Kibana.

### 6. Module Giá Vàng (6_goldprice)
- **Mô tả**: Dự báo giá vàng sử dụng dữ liệu lịch sử và mô hình học máy.
- **Tính năng**:
  - Dữ liệu giá vàng hàng ngày và hàng tháng.
  - Notebook Jupyter để phân tích và dự báo 5 năm.
  - Xuất file CSV dự báo.
- **Công nghệ**: Python, Jupyter Notebook, Pandas, Scikit-learn (dự đoán).

### 7. Mô Hình Chú Thích Hình Ảnh Bóng Đá (7_Image Captioning Model for Football Highlights)
- **Mô tả**: Mô hình học sâu để tạo chú thích cho hình ảnh highlights bóng đá.
- **Tính năng**:
  - Sử dụng ResNet50 cho trích xuất đặc trưng hình ảnh.
  - LSTM với cơ chế attention để tạo văn bản.
  - Đánh giá với BLEU, METEOR, ROUGE.
  - Demo web với Streamlit.
  - Dữ liệu: Hình ảnh và chú thích từ video bóng đá.
- **Công nghệ**: TensorFlow/Keras, OpenCV, NLTK, Streamlit, Pandas, NumPy.

## Các File Chung
- **data_sourcing.ipynb**: Notebook để thu thập dữ liệu.
- **docker-compose.yml**: Cấu hình Docker cho toàn bộ dự án.
- **OAuth 2.0 Client IDs**: Thông tin xác thực Google OAuth.
- **.gitignore**: Loại trừ file lớn như video, mô hình (.mkv, .h5).

## Kết Luận
Dự án này thể hiện sự đa dạng trong việc áp dụng công nghệ, từ API integration, web development, đến deep learning. Mỗi module có thể được phát triển độc lập hoặc tích hợp để tạo ra hệ thống lớn hơn. Dự án đã được đẩy lên GitHub tại https://github.com/Luna777247/Personal-Project.git.

**Ngày báo cáo**: 5 tháng 12, 2025</content>
<parameter name="filePath">D:\project\Personal Project\BaoCao.md