import cv2

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
if not cap.isOpened():
    print("❌ Camera not accessible")
else:
    print("✅ Camera works!")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
