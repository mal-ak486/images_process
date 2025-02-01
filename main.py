from flask import Flask, render_template, request, jsonify
import cv2
import os
import uuid
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# دالة لتطبيق الفلاتر على الصورة
def apply_filter(image, filter_type):
    if filter_type == 'grayscale':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif filter_type == 'gaussian_blur':
        return cv2.GaussianBlur(image, (15, 15), 0)
    elif filter_type == 'canny_edge':
        return cv2.Canny(image, 100, 200)
    elif filter_type == 'crop':
        height, width = image.shape[:2]
        return image[(height//2)-150:(height//2)+150, (width//2)-150:(width//2)+150]
    elif filter_type == 'resize1':
        return cv2.resize(image, (350, 350))
    elif filter_type == 'resize2':
        return cv2.resize(image, (1000, 1000))
    elif filter_type == 'rotate1':
        return cv2.flip(image, 0)
    elif filter_type == 'rotate2':
        return cv2.flip(image, 1)
    elif filter_type == 'median_blur':
        return cv2.medianBlur(image, 5)
    elif filter_type == 'bilateral_filter':
        return cv2.bilateralFilter(image, 9, 75, 75)
    elif filter_type == 'border':
        return cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif filter_type == 'hsv':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif filter_type == 'negative':
        return cv2.bitwise_not(image)
    elif filter_type == 'log_transform':
        c = 255 / np.log(1 + np.max(image))
        log_image = c * (np.log(image + 1))
        return np.uint8(log_image)
    elif filter_type == 'histogram_equalization':
        if len(image.shape) == 2:
            return cv2.equalizeHist(image)
        else:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    elif filter_type == 'average_blur':
        return cv2.blur(image, (5, 5))
    elif filter_type == 'laplacian':
        return cv2.Laplacian(image, cv2.CV_64F)
    elif filter_type == 'sobel':
        return cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)
    elif filter_type == 'difference':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, cv2.GaussianBlur(gray, (5, 5), 0))
        return diff
    elif filter_type == 'rotate_custom':
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        angle = float(request.form.get('angle', 45))
        scale = float(request.form.get('scale', 1.0))
        M = cv2.getRotationMatrix2D(center, angle, scale)
        return cv2.warpAffine(image, M, (w, h))
    elif filter_type == 'translate':
        (h, w) = image.shape[:2]
        tx = float(request.form.get('tx', 50))
        ty = float(request.form.get('ty', 50))
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, M, (w, h))
    return image

# دالة لإضافة الرسومات أو النص إلى الصورة
def apply_drawing(image, draw_type, custom_text='Hello'):
    if draw_type == 'text':
        cv2.putText(image, custom_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    elif draw_type == 'rectangle':
        cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), 2)
    elif draw_type == 'circle':
        cv2.circle(image, (150, 150), 100, (0, 0, 255), 2)
    elif draw_type == 'line':
        cv2.line(image, (50, 50), (200, 200), (255, 0, 0), 2)
    return image

# الصفحة الرئيسية
@app.route('/')
def home():
    return render_template('index.html')

# معالجة الصورة
@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filter_type = request.form.get('filter', 'none')
    draw_type = request.form.get('draw', 'none')
    custom_text = request.form.get('customText', 'Hello')  # استقبال النص من المستخدم

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # حفظ الملف المرفوع
    filename = str(uuid.uuid4()) + '.png'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # معالجة الصورة
    image = cv2.imread(filepath)
    image = apply_filter(image, filter_type)
    image = apply_drawing(image, draw_type, custom_text)  # تمرير النص إلى الدالة

    # حفظ الصورة المعالجة
    processed_filename = 'processed_' + filename
    processed_filepath = os.path.join(app.config['STATIC_FOLDER'], processed_filename)
    cv2.imwrite(processed_filepath, image)

    # إرجاع رابط الصورة المعالجة
    image_url = f"/static/{processed_filename}"
    return jsonify({'image_url': image_url})

# تشغيل التطبيق
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(STATIC_FOLDER, exist_ok=True)
    app.run(debug=True, port=5001)