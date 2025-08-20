import cv2
import numpy as np
import base64
import datetime
from flask import Blueprint, request, jsonify
from database import get_db_connection
from face_recognition_model import extract_face_encodings

attendance_bp = Blueprint("attendance", __name__)

@attendance_bp.route("/attendance", methods=["POST"])
def attendance():
    data = request.json
    image_url = data.get("image_url")

    if not image_url:
        return jsonify({"error": "Missing image_url"}), 400

    image_data = image_url.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Failed to decode image"}), 400

    face_encoding = extract_face_encodings(image)
    if face_encoding is None:
        return jsonify({"error": "No face detected"}), 400

    face_encoding = face_encoding / np.linalg.norm(face_encoding)

    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("SELECT student_id, face_encoding FROM users")
    users = cursor.fetchall()

    for student_id, face_encoding_str in users:
        stored_encoding = np.array(list(map(float, face_encoding_str.split(","))))
        stored_encoding = stored_encoding / np.linalg.norm(stored_encoding)

        distance = np.linalg.norm(face_encoding - stored_encoding)

        if distance < 1.2:
            timestamp = datetime.datetime.now()
            cursor.execute(
                "INSERT INTO attendance (student_id, timestamp, status) VALUES (%s, %s, %s)",
                (student_id, timestamp, "Present")
            )
            db.commit()

            return jsonify({
                "message": "Attendance recorded",
                "student_id": student_id,
                "timestamp": str(timestamp)
            }), 201

    return jsonify({"error": "Face not recognized"}), 400
