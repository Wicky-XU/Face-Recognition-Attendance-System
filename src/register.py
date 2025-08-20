import cv2
import numpy as np
from flask import Blueprint, request, jsonify
from database import get_db_connection
from face_recognition_model import extract_face_encodings

register_bp = Blueprint("register", __name__)

@register_bp.route('/register', methods=['POST'])
def register():
    if "image" not in request.files:
        return jsonify({"error": "Missing image file"}), 400

    image_file = request.files["image"]
    name = request.form.get("name")
    student_id = request.form.get("student_id")
    email = request.form.get("email")

    if not name or not student_id or not email:
        return jsonify({"error": "Missing required fields"}), 400

    # 读取图像并转换为 OpenCV 格式
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # 提取人脸向量（512维）
    face_encoding = extract_face_encodings(image)
    if face_encoding is None:
        return jsonify({"error": "No face detected"}), 400

    # 向量转换为字符串存储
    face_encoding_str = ",".join(map(str, face_encoding.tolist()))

    # 插入数据库
    db = get_db_connection()
    cursor = db.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (name, student_id, email, face_encoding) VALUES (%s, %s, %s, %s)",
            (name, student_id, email, face_encoding_str)
        )
        db.commit()
        return jsonify({"message": "User registered successfully!"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500
