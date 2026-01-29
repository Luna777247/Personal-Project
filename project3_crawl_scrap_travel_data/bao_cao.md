# Báo cáo Dự án Nền tảng Dữ liệu Du lịch

## Tổng quan Dự án

Dự án này là một nền tảng dữ liệu du lịch toàn diện, bao gồm:
- **Backend**: Ứng dụng FastAPI được xây dựng bằng cách sử dụng factory pattern (`core/app_factory.py`). Điểm vào chính: `main.py`, `main_frontend.py`.
- **Frontend**: Ứng dụng React SPA. Client API: `src/services/apiService.js` (proxy dev -> `http://localhost:8000`).
- **Lớp dữ liệu**: `data_platform/` chứa mã ingestion, storage và analytics (RapidAPI ingestion, Mongo helpers).

## Kiến trúc và Mô hình An toàn

### Mô hình Chỉnh sửa An toàn
- **Async-first**: Ưu tiên `async def` + `motor` cho DB trong `core/dependencies.py` và routers.
- **Import tùy chọn**: Các module bảo vệ tính năng tùy chọn bằng try/except (xem `core/app_factory.py`).
- **Hình dạng phản hồi API chuẩn**: Theo {success, status_code, data, timestamp, request_id} được sử dụng bởi các routers hiện có.

### Quy trình Phát triển Quan trọng
- Khởi động backend (dev):
  ```
  cd "web_dashboard/backend"; python main.py
  ```
- Khởi động frontend (dev):
  ```
  cd "web_dashboard/frontend"; npm install; npm start
  ```
- Launcher tích hợp: `cd "web_dashboard/backend"; python main_frontend.py integrated`
- Chạy test chính từ repo root:
  ```
  python test_rapidapi_integration.py
  python test_search_results.py
  ```

## Quy ước và Lưu ý

- **Cấu hình**: Qua `.env` và `core.config.get_config()`.
- **RapidAPI ingestion**: Ở `data_platform/ingestion/`, cần `RAPIDAPI_KEY` trong env.
- **MongoDB**: `MONGO_URI` được sử dụng; scripts thường dùng `pymongo` đồng bộ trong khi backend dùng `motor` async.
- **Entrypoints**: Nhiều `main_*.py`; tránh tạo mới trừ khi cần (ưu tiên `main.py` / `main_frontend.py`).

## Điểm Tích hợp Chính

- **Vòng đời app & routers**: `core/app_factory.py`, `web_dashboard/backend/routers/`.
- **Khởi động & env**: `main.py`, `core/config.py`.
- **DB helpers**: `core/dependencies.py`, `data_platform/storage/`.
- **Background tasks**: `routers/search.py` demo sử dụng FastAPI `BackgroundTasks`.

## Ví dụ Nhỏ

Thêm job background:
```python
from fastapi import BackgroundTasks

@router.post('/search')
async def start_search(req: SearchRequest, background_tasks: BackgroundTasks):
    search_id = f"search_{int(time.time())}"
    await storage.create_request(search_id, req.dict())
    background_tasks.add_task(process_search_request, search_id, req)
    return {"success": True, "search_id": search_id}
```

## Danh sách Kiểm tra Trước Commit

- Chạy unit/integration tests ở repo root.
- Giữ async/sync tách biệt.
- Giữ fallback behavior cho tính năng tùy chọn.
- Cập nhật `web_dashboard/frontend/src/services/apiService.js` nếu thay đổi routes hoặc response shapes.

## Cấu trúc Dự án

- Thư mục chính: `web_dashboard/backend/` (FastAPI), `web_dashboard/frontend/` (React), `data_platform/` (pipeline xử lý dữ liệu).
- Tránh files legacy: `version1/`, `version2/`, `note/`.
- Dọn dẹp artifacts: `__pycache__/`, `.pyc`, `frontend/build/`.

## Kiểm tra Sức khỏe

- Health: GET `/health` (Swagger UI tại `/docs`).
- Tests: `test_rapidapi_integration.py`, `test_search_results.py`.

Dự án này cung cấp nền tảng vững chắc cho việc quản lý và phân tích dữ liệu du lịch, với tích hợp API bên ngoài và giao diện người dùng hiện đại.