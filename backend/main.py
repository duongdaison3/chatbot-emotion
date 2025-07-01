# backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from deepface import DeepFace
import logging
import os ## THAY ĐỔI: Thêm thư viện os để đọc biến môi trường
import google.generativeai as genai ## THAY ĐỔI: Thêm thư viện của Google

# --- PHẦN MỚI: Cấu hình Gemini AI ---
try:
    # Lấy API Key từ biến môi trường
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Khởi tạo mô hình (sử dụng tên mô hình mới)
    model = genai.GenerativeModel('gemini-1.5-flash')
    logging.info("Đã cấu hình và khởi tạo mô hình Gemini AI thành công.")
except Exception as e:
    logging.error(f"Lỗi khi cấu hình Gemini AI: {e}. Hãy chắc chắn bạn đã set biến môi trường GOOGLE_API_KEY.")
    model = None
# --- KẾT THÚC PHẦN MỚI ---

logging.basicConfig(level=logging.INFO)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image: str

## THAY ĐỔI: Tạo một hàm riêng để gọi AI và lấy lời khuyên
def get_advice_from_ai(emotion: str):
    if not model:
        return "Lỗi: Mô hình AI chưa được khởi tạo. Vui lòng kiểm tra API key."
    
    # Tạo một câu lệnh (prompt) rõ ràng cho AI
    prompt = f"""
    Bạn là một trợ lý AI đồng cảm và tinh tế. Người dùng vừa được nhận diện có cảm xúc là '{emotion}'.
    Dựa vào cảm xúc này, hãy đưa ra một lời khuyên hoặc một lời động viên ngắn gọn (khoảng 1-2 câu), tích cực bằng tiếng Việt.
    - Nếu cảm xúc là 'happy' (vui vẻ), hãy chia sẻ niềm vui đó.
    - Nếu là 'sad' (buồn), hãy an ủi một cách chân thành.
    - Nếu là 'angry' (tức giận), hãy gợi ý cách giữ bình tĩnh.
    - Nếu là 'surprise' (ngạc nhiên), hãy thể hiện sự thú vị.
    - Nếu là 'neutral' (bình thường), hãy gợi ý một điều gì đó nhỏ để ngày hôm nay trở nên đặc biệt hơn.
    - Với các cảm xúc khác, hãy ứng biến một cách phù hợp.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Lỗi khi gọi Gemini API: {e}")
        return "Rất tiếc, tôi đang không thể kết nối tới bộ não AI của mình."


@app.post("/api/analyze")
async def analyze_emotion(request: ImageRequest):
    try:
        image_data_base64 = request.image.split(',')[1]
        image_data = base64.b64decode(image_data_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Dữ liệu ảnh không hợp lệ.")
        
        analysis_result = DeepFace.analyze(
            img_path=img,
            actions=['emotion'],
            enforce_detection=True
        )

        if isinstance(analysis_result, list) and len(analysis_result) > 0:
            dominant_emotion = analysis_result[0]['dominant_emotion']
            logging.info(f"Phân tích thành công. Cảm xúc chính: {dominant_emotion}")

            # ## THAY ĐỔI: Gọi hàm lấy lời khuyên từ AI ##
            advice = get_advice_from_ai(dominant_emotion)
            
            # ## THAY ĐỔI: Trả về cả cảm xúc và lời khuyên ##
            return {"emotion": dominant_emotion, "advice": advice}
        else:
             raise HTTPException(status_code=400, detail="Không thể trích xuất kết quả phân tích.")

    except ValueError:
        raise HTTPException(status_code=400, detail="Không tìm thấy khuôn mặt trong ảnh.")
    except Exception as e:
        logging.error(f"Lỗi không xác định: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi từ server: {str(e)}")