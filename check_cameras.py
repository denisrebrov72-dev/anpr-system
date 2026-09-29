import cv2

print("Проверка камер (индексы 0-5):")
for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW важен на Windows
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"  ✅ Индекс {i}: камера работает ({w}x{h})")
        else:
            print(f"  ⚠️  Индекс {i}: открылась, но кадр не читается")
        cap.release()
    else:
        print(f"  ❌ Индекс {i}: камера не найдена")