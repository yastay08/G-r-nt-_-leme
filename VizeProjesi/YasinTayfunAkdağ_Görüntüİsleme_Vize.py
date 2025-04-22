# Mediapipe ve diğer gerekli kütüphanelerin import edilmesi
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import screen_brightness_control as sbc  # Parlaklık kontrol kütüphanesi

# Görsel düzenleme parametreleri
MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # El türü yazısı için renk

# Koordinatları resim boyutuna göre dönüştürme fonksiyonu
def koordinat_getir(landmarks, indeks, h, w):
    landmark = landmarks[indeks]
    return int(landmark.x * w), int(landmark.y * h)  # Elin koordinatlarını dönüşüm yaparak geri döndürür

# Resme el izlerini çizme fonksiyonu
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks  # Elin izleri
    handedness_list = detection_result.handedness  # Elin sağ mı sol mu olduğunu belirler
    annotated_image = np.copy(rgb_image)  # Geriye çizilmiş resim
    h, w, c = annotated_image.shape  # Resmin boyutları (yükseklik, genişlik)

    toplam_acik = None  # Başlangıçta parmak sayısı

    # Her bir eldeki izleri döngü ile işleme
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]  # Elin izleri
        parmaklar = []  # Parmağın açık olup olmadığını tutacak liste

        # İşaret parmağı kontrolü (baş parmak ile kıyaslama)
        x1, y1 = koordinat_getir(hand_landmarks, 8, h, w)
        x5, y5 = koordinat_getir(hand_landmarks, 6, h, w)
        parmaklar.append(1 if y1 < y5 else 0)

        # Orta parmak kontrolü
        x1, y1 = koordinat_getir(hand_landmarks, 12, h, w)
        x5, y5 = koordinat_getir(hand_landmarks, 10, h, w)
        parmaklar.append(1 if y1 < y5 else 0)

        # Yüzük parmak kontrolü
        x1, y1 = koordinat_getir(hand_landmarks, 16, h, w)
        x5, y5 = koordinat_getir(hand_landmarks, 14, h, w)
        parmaklar.append(1 if y1 < y5 else 0)

        # Serçe parmak kontrolü
        x1, y1 = koordinat_getir(hand_landmarks, 20, h, w)
        x5, y5 = koordinat_getir(hand_landmarks, 18, h, w)
        parmaklar.append(1 if y1 < y5 else 0)

        # Baş parmak kontrolü (el ayasına uzaklık)
        x0, y0 = koordinat_getir(hand_landmarks, 0, h, w)  # Elin iç kısmı
        x4, y4 = koordinat_getir(hand_landmarks, 4, h, w)  # Baş parmak ucu
        bas_parmak_mesafe = abs(x4 - x0)  # Baş parmak ile elin iç kısmı arasındaki mesafe
        parmaklar.append(1 if bas_parmak_mesafe > 40 else 0)  # Eğer mesafe 40'dan büyükse baş parmak açık sayılır

        toplam_acik = sum(parmaklar)  # Açık parmakların sayısı

        # Görsel işaretleme (parmak pozisyonu ve sayısı)
        x1, y1 = koordinat_getir(hand_landmarks, 8, h, w)
        annotated_image = cv2.circle(annotated_image, (x1, y1), 9, (255, 255, 0), 5)  # Parmağa yuvarlak çiz
        annotated_image = cv2.putText(annotated_image, str(toplam_acik), (x1, y1),
                                      cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 4)  # Parmak sayısını yaz

        # Elin izlerini çizme (MediaPipe'in yardımcı fonksiyonu ile)
        handedness = handedness_list[idx]  # Elin sağ mı sol mu olduğu bilgisi
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
            solutions.drawing_styles.get_default_hand_connections_style())  # El izlerini çiz

        # El türünü (sol/sağ) yazdırma
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * w)
        text_y = int(min(y_coordinates) * h) - MARGIN

        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)  # El türünü yaz

    return annotated_image, toplam_acik  # İşaretli resim ve toplam açık parmak sayısını döndür

# Mediapipe ile model yükleme ve ayarların yapılması
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# El izleyici modelini yüklemek için ayar
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')  # Model dosyasını belirtiyoruz
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)  # Maksimum 2 el algılayacak şekilde ayar yapıyoruz
detector = vision.HandLandmarker.create_from_options(options)  # El izleyiciyi oluşturuyoruz

# Webcam açma
cam = cv2.VideoCapture(0)  # Kamera (webcam) bağlantısını açıyoruz

# Sonsuz döngü ile görüntü alma ve işlem yapma
while cam.isOpened():
    basari, frame = cam.read()  # Kameradan bir görüntü al
    if basari:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Görüntüyü RGB'ye dönüştür
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)  # MediaPipe için uygun formata dönüştür
        detection_result = detector.detect(mp_image)  # El izlerini tespit et

        # El izlerini çizme ve parmak sayısını al
        annotated_image, toplam_acik = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)

        # Parmak sayısına göre ekran parlaklığını ayarla
        if toplam_acik is not None:
            parlaklik = {
                0: 0,  # 0 parmak açık, parlaklık en düşük
                1: 20,  # 1 parmak açık, parlaklık biraz daha düşük
                2: 40,  # 2 parmak açık, parlaklık ortalama
                3: 60,  # 3 parmak açık, parlaklık yüksek
                4: 80,  # 4 parmak açık, parlaklık çok yüksek
                5: 100  # 5 parmak açık, parlaklık en yüksek
            }.get(toplam_acik, 100)  # Parmak sayısına göre parlaklık ayarla
            sbc.set_brightness(parlaklik)  # Ekran parlaklığını ayarla

        # Ekranda resmi göster
        cv2.imshow("Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))  # Görüntüyü ekranda göster

        # 'q' tuşuna basıldığında döngüden çık
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):  # 'q' tuşu ile çık
            break
