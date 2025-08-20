let cameraStream = null;

// 用户注册（提交姓名、学号、邮箱和照片）
function registerUser() {
    let formData = new FormData();
    formData.append("name", document.getElementById("name").value);
    formData.append("student_id", document.getElementById("student_id").value);
    formData.append("email", document.getElementById("email").value);
    formData.append("image", document.getElementById("image").files[0]);

    fetch("/register", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message || data.error);
    })
    .catch(error => console.error("Error:", error));
}

// 更新文件名显示（防止上传按钮显示为中文）
function updateFileName() {
    let fileInput = document.getElementById("image");
    let fileNameDisplay = document.getElementById("fileName");
    fileNameDisplay.innerText = fileInput.files.length > 0 ? fileInput.files[0].name : "No file selected";
}


// 切换摄像头
function toggleCamera() {
    let video = document.getElementById("camera");
    let captureButton = document.getElementById("captureButton");
    let uploadButton = document.getElementById("uploadButton");
    let toggleButton = document.getElementById("toggleCamera");

    if (cameraStream) {
        // 关闭摄像头
        let tracks = cameraStream.getTracks();
        tracks.forEach(track => track.stop());
        cameraStream = null;
        video.style.display = "none";
        captureButton.style.display = "none";
        uploadButton.style.display = "none";
        toggleButton.innerText = "Turn Camera On";
    } else {
        // 打开摄像头
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                cameraStream = stream;
                video.srcObject = stream;
                video.style.display = "block";
                captureButton.style.display = "inline-block";
                uploadButton.style.display = "inline-block";
                toggleButton.innerText = "Turn Camera Off";
            })
            .catch(error => console.error("Error accessing webcam:", error));
    }
}

// 拍照
function capturePhoto() {
    let video = document.getElementById("camera");
    let canvas = document.getElementById("canvas");
    let context = canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.style.display = "block";
}

// 上传考勤数据（仅限摄像头拍照）
function uploadAttendance() {
    let canvas = document.getElementById("canvas");
    let imageData = canvas.toDataURL("image/jpeg");

    fetch("/attendance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_url: imageData })
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message || data.error);
    })
    .catch(error => console.error("Error:", error));
}


function exportAttendance() {
    fetch('/export-attendance')
        .then(response => response.blob())
        .then(blob => {
            let a = document.createElement("a");
            a.href = window.URL.createObjectURL(blob);
            a.download = "attendance_records.xlsx";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        })
        .catch(error => console.error("Error exporting data:", error));
}
