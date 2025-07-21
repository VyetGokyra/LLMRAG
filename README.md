# RAG-based Question Answering System

Hệ thống trả lời câu hỏi sử dụng kỹ thuật RAG (Retrieval Augmented Generation) với việc lưu trữ và tìm kiếm thông tin từ nhiều nguồn dữ liệu khác nhau.

## Yêu cầu

- Python 3.8 trở lên
- OpenAI API key
- Virtual environment (venv hoặc conda)

## Cài đặt

1. Clone repository và tạo môi trường ảo:
```bash
git clone <repository-url>
cd LLMRAG
python -m venv venv
source venv/bin/activate  # Trên Linux/Mac
# hoặc
.\venv\Scripts\activate  # Trên Windows
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

3. Cấu hình OpenAI API key:
```bash
# Tạo file .env trong thư mục gốc của project
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## Chuẩn bị dữ liệu

Dữ liệu nguồn được đặt trong thư mục `data_sources/` với cấu trúc như sau:
```
data_sources/
    csv/
        books.csv
    json_source/
        books.json
    pdf_source/
        book1.pdf
        book2.pdf
```

## Chạy ứng dụng

1. Khởi động server FastAPI:
```bash
uvicorn server:app
```

2. Server sẽ tự động:
- Đọc và xử lý tất cả dữ liệu từ thư mục `data_sources/`
- Tạo embeddings và lưu vào ChromaDB
- Khởi tạo Retriever và LLM chain

3. Truy cập giao diện API:
- Mở trình duyệt và truy cập http://127.0.0.1:8000/docs
- Sử dụng endpoint `/chat` để gửi câu hỏi
- Cần cung cấp 2 tham số:
  - `question`: Câu hỏi cần trả lời
  - `session_id`: ID phiên chat (có thể là bất kỳ chuỗi nào, ví dụ: "123")

## Ví dụ sử dụng

1. Thông qua giao diện Swagger UI:
- Truy cập http://127.0.0.1:8000/docs
- Chọn endpoint `/chat`
- Click "Try it out"
- Nhập câu hỏi và session_id
- Click "Execute"

2. Thông qua HTTP request:
```bash
curl -X GET "http://127.0.0.1:8000/chat?question=thám%20tử%20conan%20của%20nhà%20xuất%20bản%20nào?&session_id=123"
```

## Lưu ý

- Mỗi session_id sẽ duy trì một lịch sử chat riêng
- Giữ nguyên session_id nếu muốn tiếp tục cuộc trò chuyện
- Sử dụng session_id mới nếu muốn bắt đầu cuộc trò chuyện mới
- Embeddings và dữ liệu được lưu trong thư mục `chroma_db/`
- Có thể xóa thư mục `chroma_db/` để tạo lại vector store từ đầu
