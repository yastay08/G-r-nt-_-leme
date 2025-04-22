from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import screen_brightness_control as sbc  # parlaklık kontrol kütüphanesi

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

def koordinat_getir(landmarks, indeks, h, w):
    landmark = landmarks[indeks]
    return int(landmark.x * w), int(landmark.y * h)

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    h, w, c = annotated_image.shape

    toplam_acik = None

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        parmaklar = []

        # İşaret parmağı
        x1, y1 = koordinat_getir(hand_landmarks, 8, h, w)
        x5, y5 = koordinat_getir(hand_landmarks, 6, h, w)
        parmaklar.append(1 if y1 < y5 else 0)

        # Orta parmak
        x1, y1 = koordinat_getir(hand_landmarks, 12, h, w)
        x5, y5 = koordinat_getir(hand_landmarks, 10, h, w)
        parmaklar.append(1 if y1 < y5 else 0)

        # Yüzük parmak
        x1, y1 = koordinat_getir(hand_landmarks, 16, h, w)
        x5, y5 = koordinat_getir(hand_landmarks, 14, h, w)
        parmaklar.append(1 if y1 < y5 else 0)

        # Serçe parmak
        x1, y1 = koordinat_getir(hand_landmarks, 20, h, w)
        x5, y5 = koordinat_getir(hand_landmarks, 18, h, w)
        parmaklar.append(1 if y1 < y5 else 0)

        # Baş parmak (yeni yöntem: el ayasına uzaklığına göre kontrol)
        x0, y0 = koordinat_getir(hand_landmarks, 0, h, w)  # El ayası
        x4, y4 = koordinat_getir(hand_landmarks, 4, h, w)  # Baş parmak ucu
        bas_parmak_mesafe = abs(x4 - x0)
        parmaklar.append(1 if bas_parmak_mesafe > 40 else 0)  # Eşik değeri ayarlanabilir

        toplam_acik = sum(parmaklar)

        # Görsel işaretleme
        x1, y1 = koordinat_getir(hand_landmarks, 8, h, w)
        annotated_image = cv2.circle(annotated_image, (x1, y1), 9, (255, 255, 0), 5)
        annotated_image = cv2.putText(annotated_image, str(toplam_acik), (x1, y1),
                                      cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 4)

        # El çizimi
        handedness = handedness_list[idx]
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in hand_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # El türü yazısı (Left / Right)
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * w)
        text_y = int(min(y_coordinates) * h) - MARGIN

        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image, toplam_acik

# Mediapipe yüklemesi
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

cam = cv2.VideoCapture(0)

while cam.isOpened():
    basari, frame = cam.read()
    if basari:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_image)

        annotated_image, toplam_acik = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)

        # Parmak sayısına göre ekran parlaklığı ayarla
        if toplam_acik is not None:
            parlaklik = {
                0: 0,
                1: 20,
                2: 40,
                3: 60,
                4: 80,
                5: 100
            }.get(toplam_acik, 100)
            sbc.set_brightness(parlaklik)

        cv2.imshow("Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break
