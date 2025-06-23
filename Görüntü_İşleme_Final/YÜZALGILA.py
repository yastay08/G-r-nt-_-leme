#@markdown We implemented some functions to visualize the face landmark detection results. <br/> Run the following cell to activate the functions.
from mediapipe import solutions
import numpy as np
import matplotlib.pyplot as plt
import cv2

def draw_landmarks_on_image(rgb_goruntu, tespit_sonucu):
    yuz_noktalar_listesi = tespit_sonucu.face_landmarks
    islenmis_goruntu = np.copy(rgb_goruntu)

    for yuz_noktalari in yuz_noktalar_listesi:
        # 📌 Yüzü mozaikle (KVKK uyumu)
        kordinat_x = [n.x for n in yuz_noktalari]
        kordinat_y = [n.y for n in yuz_noktalari]
        yukseklik, genislik, _ = islenmis_goruntu.shape
        x_enaz = int(min(kordinat_x) * genislik)
        x_encok = int(max(kordinat_x) * genislik)
        y_enaz = int(min(kordinat_y) * yukseklik)
        y_encok = int(max(kordinat_y) * yukseklik)

        bolge = islenmis_goruntu[y_enaz:y_encok, x_enaz:x_encok]
        if bolge.size > 0:
            kucultulmus = cv2.resize(bolge, (16, 16), interpolation=cv2.INTER_LINEAR)
            mozaik = cv2.resize(kucultulmus, (x_encok - x_enaz, y_encok - y_enaz), interpolation=cv2.INTER_NEAREST)
            islenmis_goruntu[y_enaz:y_encok, x_enaz:x_encok] = mozaik

    return islenmis_goruntu


def plot_face_blendshapes_bar_graph(face_blendshapes):
    isimler = [b.category_name for b in face_blendshapes]
    skorlar = [b.score for b in face_blendshapes]
    siralama = range(len(isimler))

    fig, ax = plt.subplots(figsize=(12, 12))
    cubuklar = ax.barh(siralama, skorlar)
    ax.set_yticks(siralama, isimler)
    ax.invert_yaxis()

    for skor, cubuk in zip(skorlar, cubuklar.patches):
        plt.text(cubuk.get_x() + cubuk.get_width(), cubuk.get_y(), f"{skor:.4f}", va="top")

    ax.set_xlabel('Skor')
    ax.set_title("Yüz İfadeleri")
    plt.tight_layout()
    plt.show()


# STEP 1: Import mediapipe and setup
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

temel_ayarlar = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
ayarlar = vision.FaceLandmarkerOptions(
    base_options=temel_ayarlar,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
tespitci = vision.FaceLandmarker.create_from_options(ayarlar)

# STEP 2: Kamera başlat
kamera = cv2.VideoCapture(0)
while kamera.isOpened():
    basarili, goruntu = kamera.read()
    if basarili:
        goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)
        mp_goruntu = mp.Image(image_format=mp.ImageFormat.SRGB, data=goruntu)

        sonuc = tespitci.detect(mp_goruntu)

        islenmis = draw_landmarks_on_image(mp_goruntu.numpy_view(), sonuc)
        cv2.imshow("yuz", cv2.cvtColor(islenmis, cv2.COLOR_RGB2BGR))
        tus = cv2.waitKey(1)
        if tus == ord('q') or tus == ord('Q'):
            break

kamera.release()
cv2.destroyAllWindows()