import cv2
import mediapipe as mp

# Inisialisasi Face Detection dari Mediapipe
mp_face_detection = mp.solutions.face_detection

# Buka webcam
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Konversi warna untuk MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # Gambar kotak kuning di wajah
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 255), 2)

        # Tampilkan hasil
        cv2.imshow('Webcam Face Detection', frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Tutup webcam
cap.release()
cv2.destroyAllWindows()
