import os
from flask import Flask, render_template, send_file
from database import get_db_connection
from register import register_bp
from attendance import attendance_bp
import pandas as pd
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter


# Disable TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__, static_folder='static')

# Register API map
app.register_blueprint(register_bp)
app.register_blueprint(attendance_bp)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register-page')
def register_page():
    return render_template('register.html')

@app.route('/attendance-page')
def attendance_page():
    return render_template('attendance.html')

@app.route('/records')
def records():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT student_id, timestamp, status FROM attendance ORDER BY timestamp DESC")
    records = cursor.fetchall()
    return render_template('records.html', records=records)


@app.route('/export-attendance')
def export_attendance():
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("SELECT student_id, timestamp, status FROM attendance")
    data = cursor.fetchall()

    # Creat Pandas DataFrame
    df = pd.DataFrame(data, columns=['Student ID', 'Timestamp', 'Status'])

    # Generate Excel Document
    file_path = "attendance_records.xlsx"
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Attendance Records")
        workbook = writer.book
        worksheet = writer.sheets["Attendance Records"]

        # Adjust width
        for col_num, col_cells in enumerate(worksheet.columns, start=1):
            max_length = max(len(str(cell.value)) if cell.value else 10 for cell in col_cells)
            adjusted_width = max_length + 5
            worksheet.column_dimensions[get_column_letter(col_num)].width = adjusted_width

        # Centered alignment
        for col in worksheet.columns:
            for cell in col:
                cell.alignment = Alignment(horizontal='center', vertical='center')

    return send_file(file_path, as_attachment=True, download_name="attendance_records.xlsx")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

