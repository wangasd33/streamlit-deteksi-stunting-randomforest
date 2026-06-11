import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import copy

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception:
        return ""

# ============================================================
# FUNGSI OVERLAY TITIK PADA GRAFIK WHO
# ============================================================
# Kalibrasi pixel berdasarkan posisi gridline pada gambar grafik WHO
# Sumbu X: Birth (0 bulan) — 5 years (60 bulan)
# Sumbu Y: 45 cm (bawah) — 125 cm (atas)
CHART_CALIBRATION = {
    "Laki-laki": {
        "image": "assets/images/Grafik anak laki-laki - Panjang atau tinggi menurut usia dari lahir hingga 5 tahun (skor z)_page-0001.jpg",
        "birth_x": 503.0,    # pixel x untuk usia 0 bulan
        "ppm": 42.0,         # pixels per month
        "y_45": 2016.0,      # pixel y untuk 45 cm
        "ppc": 18.96,        # pixels per cm
    },
    "Perempuan": {
        "image": "assets/images/Grafik anak perempuan - Panjang atau tinggi badan menurut usia dari lahir hingga 5 tahun (skor z)_page-0001.jpg",
        "birth_x": 503.0,    # pixel x untuk usia 0 bulan
        "ppm": 42.0,         # pixels per month
        "y_45": 2016.0,      # pixel y untuk 45 cm
        "ppc": 18.96,        # pixels per cm
    }
}

def create_growth_chart_with_marker(jenis_kelamin, usia_bulan, tinggi_cm, prediksi_label):
    """
    Membuat gambar grafik WHO dengan overlay titik posisi anak.
    
    Args:
        jenis_kelamin: "Laki-laki" atau "Perempuan"
        usia_bulan: usia anak dalam bulan (0-60)
        tinggi_cm: tinggi badan anak dalam cm
        prediksi_label: hasil prediksi ("Normal", "Stunted", "Severely Stunted")
    
    Returns:
        PIL Image object dengan titik ter-overlay
    """
    cal = CHART_CALIBRATION[jenis_kelamin]
    
    # Buka gambar grafik
    img = Image.open(cal["image"]).convert("RGBA")
    
    # Hitung posisi pixel
    px = cal["birth_x"] + usia_bulan * cal["ppm"]
    py = cal["y_45"] - (tinggi_cm - 45) * cal["ppc"]
    
    # Pastikan titik berada dalam batas gambar
    px = max(0, min(px, img.width - 1))
    py = max(0, min(py, img.height - 1))
    
    # Warna titik berdasarkan prediksi
    if prediksi_label == "Normal":
        dot_color = (34, 197, 94, 255)       # Hijau
        outline_color = (21, 128, 61, 255)
        label_text = "Normal"
    elif prediksi_label == "Stunted":
        dot_color = (251, 146, 60, 255)      # Oranye
        outline_color = (194, 65, 12, 255)
        label_text = "Stunted"
    else:
        dot_color = (239, 68, 68, 255)       # Merah
        outline_color = (185, 28, 28, 255)
        label_text = "Severely Stunted"
    
    # Buat overlay transparan
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Gambar titik besar (lingkaran) dengan outline
    dot_radius = 20
    outline_width = 5
    
    # Outer ring (outline)
    draw.ellipse(
        [px - dot_radius - outline_width, py - dot_radius - outline_width,
         px + dot_radius + outline_width, py + dot_radius + outline_width],
        fill=outline_color
    )
    
    # Inner circle (main dot)
    draw.ellipse(
        [px - dot_radius, py - dot_radius, px + dot_radius, py + dot_radius],
        fill=dot_color
    )
    
    # Titik putih kecil di tengah (highlight)
    highlight_r = 6
    draw.ellipse(
        [px - highlight_r, py - highlight_r,
         px + highlight_r, py + highlight_r],
        fill=(255, 255, 255, 180)
    )
    
    # Label teks di samping titik
    try:
        font = ImageFont.truetype("arial.ttf", 36)
        font_bold = ImageFont.truetype("arialbd.ttf", 36)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        except (IOError, OSError):
            font = ImageFont.load_default()
            font_bold = font
    
    # Posisi label (di kanan atas titik)
    label_x = px + dot_radius + 15
    label_y = py - dot_radius - 20
    
    # Jika label terlalu ke kanan, pindah ke kiri
    if label_x + 300 > img.width:
        label_x = px - dot_radius - 320
    
    # Jika label terlalu ke atas, pindah ke bawah
    if label_y < 20:
        label_y = py + dot_radius + 10
    
    # Background label
    label_info = f"{tinggi_cm} cm, {usia_bulan} bln"
    
    # Hitung ukuran teks
    bbox1 = draw.textbbox((0, 0), label_text, font=font_bold)
    bbox2 = draw.textbbox((0, 0), label_info, font=font)
    text_w = max(bbox1[2] - bbox1[0], bbox2[2] - bbox2[0]) + 16
    text_h = (bbox1[3] - bbox1[1]) + (bbox2[3] - bbox2[1]) + 18
    
    # Rounded rectangle background
    bg_x1 = label_x - 4
    bg_y1 = label_y - 4
    bg_x2 = bg_x1 + text_w
    bg_y2 = bg_y1 + text_h
    
    draw.rounded_rectangle([bg_x1, bg_y1, bg_x2, bg_y2], radius=6, fill=(255, 255, 255, 220), outline=outline_color, width=2)
    
    # Teks label
    draw.text((label_x + 4, label_y), label_text, fill=outline_color, font=font_bold)
    draw.text((label_x + 4, label_y + bbox1[3] - bbox1[1] + 4), label_info, fill=(60, 60, 60, 255), font=font)
    
    # Garis penunjuk dari titik ke label (opsional, garis putus-putus)
    # Composite overlay onto original image
    result = Image.alpha_composite(img, overlay)
    
    return result.convert("RGB")

icon_base64 = get_base64_image("assets/images/icon-balita.jpg")
icon_html = f"<img src='data:image/jpeg;base64,{icon_base64}' width='45' style='vertical-align: middle; margin-right: 10px; margin-bottom: 5px; border-radius: 50%;'>" if icon_base64 else "👶 "

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Deteksi Dini Stunting",
    page_icon=Image.open("assets/images/icon-balita.jpg") if os.path.exists("assets/images/icon-balita.jpg") else "👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD MODEL & MAPPING
# ============================================================
@st.cache_resource
def load_model():
    model         = joblib.load("models/rf_stunting_model.pkl")
    mapping_label = joblib.load("models/mapping_label.pkl")
    mapping_jk    = joblib.load("models/mapping_jk.pkl")
    inverse_label = {v: k for k, v in mapping_label.items()}
    inverse_jk    = {v: k for k, v in mapping_jk.items()}
    return model, mapping_label, mapping_jk, inverse_label, inverse_jk

model, mapping_label, mapping_jk, inverse_label, inverse_jk = load_model()

# ============================================================
# DATA STANDAR WHO (Height-for-Age & Weight-for-Age)
# Sumber: WHO Child Growth Standards 2006
# Format: usia_bulan: (median, -2SD, -3SD)
# ============================================================
WHO_HEIGHT = {
    "Laki-laki": {
        0: (49.9, 44.2, 42.5), 1: (54.7, 48.9, 47.2), 2: (58.4, 52.4, 50.7),
        3: (61.4, 55.3, 53.5), 4: (63.9, 57.6, 55.8), 5: (65.9, 59.6, 57.8),
        6: (67.6, 61.2, 59.4), 7: (69.2, 62.7, 60.8), 8: (70.6, 64.0, 62.1),
        9: (72.0, 65.2, 63.3), 10: (73.3, 66.4, 64.4), 11: (74.5, 67.6, 65.5),
        12: (75.7, 68.6, 66.6), 13: (76.9, 69.6, 67.6), 14: (78.0, 70.6, 68.5),
        15: (79.1, 71.6, 69.5), 16: (80.2, 72.5, 70.4), 17: (81.2, 73.5, 71.3),
        18: (82.3, 74.4, 72.2), 19: (83.2, 75.3, 73.1), 20: (84.2, 76.2, 74.0),
        21: (85.1, 77.1, 74.8), 22: (86.0, 78.0, 75.7), 23: (86.9, 78.8, 76.5),
        24: (87.8, 79.7, 77.3), 25: (88.0, 79.9, 77.5), 26: (88.8, 80.8, 78.3),
        27: (89.6, 81.5, 79.1), 28: (90.4, 82.3, 79.9), 29: (91.2, 83.1, 80.6),
        30: (91.9, 83.8, 81.4), 31: (92.7, 84.5, 82.1), 32: (93.4, 85.3, 82.8),
        33: (94.1, 86.0, 83.5), 34: (94.8, 86.7, 84.2), 35: (95.4, 87.4, 84.9),
        36: (96.1, 88.0, 85.5), 37: (96.7, 88.7, 86.2), 38: (97.4, 89.3, 86.8),
        39: (98.0, 89.9, 87.4), 40: (98.6, 90.5, 88.0), 41: (99.2, 91.1, 88.6),
        42: (99.9, 91.7, 89.2), 43: (100.4, 92.3, 89.8), 44: (101.0, 92.9, 90.3),
        45: (101.6, 93.5, 90.9), 46: (102.2, 94.0, 91.4), 47: (102.8, 94.6, 92.0),
        48: (103.3, 95.1, 92.5), 49: (103.9, 95.7, 93.1), 50: (104.4, 96.2, 93.6),
        51: (105.0, 96.8, 94.1), 52: (105.5, 97.3, 94.7), 53: (106.1, 97.8, 95.2),
        54: (106.6, 98.4, 95.7), 55: (107.2, 98.9, 96.2), 56: (107.7, 99.4, 96.7),
        57: (108.2, 99.9, 97.2), 58: (108.8, 100.5, 97.7), 59: (109.3, 101.0, 98.2),
        60: (109.8, 101.5, 98.7),
    },
    "Perempuan": {
        0: (49.1, 43.6, 41.9), 1: (53.7, 47.8, 46.1), 2: (57.1, 51.0, 49.2),
        3: (59.8, 53.5, 51.7), 4: (62.1, 55.6, 53.7), 5: (64.0, 57.4, 55.4),
        6: (65.7, 58.9, 56.9), 7: (67.3, 60.3, 58.3), 8: (68.7, 61.7, 59.6),
        9: (70.1, 62.9, 60.8), 10: (71.5, 64.1, 62.0), 11: (72.8, 65.2, 63.1),
        12: (74.0, 66.3, 64.2), 13: (75.2, 67.3, 65.2), 14: (76.4, 68.3, 66.1),
        15: (77.5, 69.3, 67.1), 16: (78.6, 70.2, 68.0), 17: (79.7, 71.1, 68.9),
        18: (80.7, 72.0, 69.8), 19: (81.7, 72.9, 70.6), 20: (82.7, 73.7, 71.4),
        21: (83.7, 74.5, 72.2), 22: (84.6, 75.4, 73.0), 23: (85.5, 76.2, 73.8),
        24: (86.4, 77.0, 74.6), 25: (86.6, 77.2, 74.8), 26: (87.4, 78.1, 75.6),
        27: (88.3, 78.9, 76.4), 28: (89.1, 79.7, 77.2), 29: (89.9, 80.5, 78.0),
        30: (90.7, 81.3, 78.7), 31: (91.4, 82.0, 79.5), 32: (92.2, 82.7, 80.2),
        33: (92.9, 83.5, 80.9), 34: (93.6, 84.2, 81.6), 35: (94.4, 84.9, 82.3),
        36: (95.1, 85.6, 83.0), 37: (95.7, 86.2, 83.6), 38: (96.4, 86.9, 84.3),
        39: (97.1, 87.5, 84.9), 40: (97.7, 88.2, 85.6), 41: (98.4, 88.8, 86.2),
        42: (99.0, 89.4, 86.8), 43: (99.7, 90.1, 87.4), 44: (100.3, 90.7, 88.0),
        45: (100.9, 91.3, 88.6), 46: (101.5, 91.9, 89.2), 47: (102.1, 92.5, 89.8),
        48: (102.7, 93.1, 90.4), 49: (103.3, 93.6, 91.0), 50: (103.9, 94.2, 91.5),
        51: (104.5, 94.8, 92.1), 52: (105.0, 95.4, 92.7), 53: (105.6, 95.9, 93.2),
        54: (106.2, 96.5, 93.8), 55: (106.7, 97.0, 94.3), 56: (107.3, 97.6, 94.9),
        57: (107.8, 98.1, 95.4), 58: (108.4, 98.7, 95.9), 59: (108.9, 99.2, 96.5),
        60: (109.4, 99.7, 97.0),
    }
}

WHO_WEIGHT = {
    "Laki-laki": {
        0: (3.3, 2.5, 2.1), 1: (4.5, 3.4, 2.9), 2: (5.6, 4.3, 3.8),
        3: (6.4, 5.0, 4.4), 4: (7.0, 5.6, 4.9), 5: (7.5, 6.0, 5.3),
        6: (7.9, 6.4, 5.7), 7: (8.3, 6.7, 5.9), 8: (8.6, 6.9, 6.2),
        9: (8.9, 7.1, 6.4), 10: (9.2, 7.4, 6.6), 11: (9.4, 7.6, 6.8),
        12: (9.6, 7.7, 6.9), 13: (9.9, 7.9, 7.1), 14: (10.1, 8.1, 7.2),
        15: (10.3, 8.3, 7.4), 16: (10.5, 8.4, 7.5), 17: (10.7, 8.6, 7.7),
        18: (10.9, 8.8, 7.8), 19: (11.1, 8.9, 7.9), 20: (11.3, 9.1, 8.1),
        21: (11.5, 9.2, 8.2), 22: (11.8, 9.4, 8.4), 23: (12.0, 9.5, 8.5),
        24: (12.2, 9.7, 8.6), 25: (12.4, 9.8, 8.8), 26: (12.5, 10.0, 8.9),
        27: (12.7, 10.1, 9.0), 28: (12.9, 10.2, 9.1), 29: (13.1, 10.4, 9.2),
        30: (13.3, 10.5, 9.4), 31: (13.5, 10.7, 9.5), 32: (13.7, 10.8, 9.6),
        33: (13.8, 10.9, 9.7), 34: (14.0, 11.1, 9.9), 35: (14.2, 11.2, 10.0),
        36: (14.3, 11.3, 10.1), 37: (14.5, 11.5, 10.2), 38: (14.7, 11.6, 10.3),
        39: (14.8, 11.7, 10.4), 40: (15.0, 11.8, 10.5), 41: (15.2, 12.0, 10.7),
        42: (15.3, 12.1, 10.8), 43: (15.5, 12.2, 10.9), 44: (15.7, 12.4, 11.0),
        45: (15.8, 12.5, 11.1), 46: (16.0, 12.6, 11.2), 47: (16.2, 12.8, 11.3),
        48: (16.3, 12.9, 11.5), 49: (16.5, 13.0, 11.6), 50: (16.7, 13.2, 11.7),
        51: (16.8, 13.3, 11.8), 52: (17.0, 13.4, 11.9), 53: (17.2, 13.6, 12.0),
        54: (17.3, 13.7, 12.2), 55: (17.5, 13.8, 12.3), 56: (17.7, 14.0, 12.4),
        57: (17.8, 14.1, 12.5), 58: (18.0, 14.2, 12.6), 59: (18.2, 14.4, 12.7),
        60: (18.3, 14.5, 12.9),
    },
    "Perempuan": {
        0: (3.2, 2.4, 2.0), 1: (4.2, 3.2, 2.7), 2: (5.1, 3.9, 3.4),
        3: (5.8, 4.5, 3.9), 4: (6.4, 5.0, 4.4), 5: (6.9, 5.4, 4.8),
        6: (7.3, 5.7, 5.1), 7: (7.6, 6.0, 5.3), 8: (7.9, 6.3, 5.6),
        9: (8.2, 6.5, 5.8), 10: (8.5, 6.7, 5.9), 11: (8.7, 6.9, 6.1),
        12: (8.9, 7.0, 6.3), 13: (9.2, 7.2, 6.4), 14: (9.4, 7.4, 6.6),
        15: (9.6, 7.6, 6.7), 16: (9.8, 7.7, 6.9), 17: (10.0, 7.9, 7.0),
        18: (10.2, 8.1, 7.2), 19: (10.4, 8.2, 7.3), 20: (10.6, 8.4, 7.5),
        21: (10.9, 8.6, 7.6), 22: (11.1, 8.7, 7.8), 23: (11.3, 8.9, 7.9),
        24: (11.5, 9.0, 8.1), 25: (11.7, 9.2, 8.2), 26: (11.9, 9.4, 8.4),
        27: (12.1, 9.5, 8.5), 28: (12.3, 9.7, 8.6), 29: (12.5, 9.8, 8.8),
        30: (12.7, 10.0, 8.9), 31: (12.9, 10.1, 9.0), 32: (13.1, 10.3, 9.1),
        33: (13.3, 10.4, 9.3), 34: (13.5, 10.5, 9.4), 35: (13.7, 10.7, 9.5),
        36: (13.9, 10.8, 9.6), 37: (14.0, 11.0, 9.7), 38: (14.2, 11.1, 9.9),
        39: (14.4, 11.2, 10.0), 40: (14.6, 11.4, 10.1), 41: (14.8, 11.5, 10.2),
        42: (15.0, 11.6, 10.4), 43: (15.2, 11.8, 10.5), 44: (15.3, 11.9, 10.6),
        45: (15.5, 12.1, 10.7), 46: (15.7, 12.2, 10.9), 47: (15.9, 12.3, 11.0),
        48: (16.1, 12.5, 11.1), 49: (16.3, 12.6, 11.2), 50: (16.4, 12.8, 11.4),
        51: (16.6, 12.9, 11.5), 52: (16.8, 13.1, 11.6), 53: (17.0, 13.2, 11.7),
        54: (17.2, 13.3, 11.9), 55: (17.3, 13.5, 12.0), 56: (17.5, 13.6, 12.1),
        57: (17.7, 13.7, 12.3), 58: (17.9, 13.9, 12.4), 59: (18.0, 14.0, 12.5),
        60: (18.2, 14.2, 12.6),
    }
}

# ============================================================
# SIDEBAR NAVIGASI
# ============================================================
with st.sidebar:
    st.image("assets/images/logo-balita.jpg", width=100)
    st.markdown(f"## {icon_html} Deteksi Stunting", unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio(
        "Pilih Menu",
        ["🏠 Beranda",
         "🔍 Prediksi Stunting"]
    )

# ============================================================
# HALAMAN: BERANDA
# ============================================================
if menu == "🏠 Beranda":

    # ---- HERO SECTION ----
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0c4a6e 100%);
                border-radius: 16px; padding: 48px 36px; text-align: center; margin-bottom: 24px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.25);">
        <p style="font-size: 48px; margin: 0;">👶</p>
        <h1 style="color: #f0f9ff; margin: 12px 0 8px 0; font-size: 32px; font-weight: 800; letter-spacing: -0.5px;">
            Sistem Deteksi Dini Stunting
        </h1>
        <p style="color: #7dd3fc; margin: 0 0 6px 0; font-size: 16px; font-weight: 500;">
            Berbasis Machine Learning — Algoritma Random Forest
        </p>
        <p style="color: #94a3b8; margin: 0; font-size: 13px;">
            Mengacu pada WHO Child Growth Standards 2006
        </p>
        <div style="margin-top: 20px;">
            <span style="background: rgba(255,255,255,0.12); color: #e0f2fe; padding: 6px 16px;
                         border-radius: 20px; font-size: 12px; font-weight: 600; letter-spacing: 0.5px;">
                🎓 PENELITIAN SKRIPSI
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- SELAMAT DATANG ----
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ecfdf5, #f0fdf4); border-left: 5px solid #22c55e;
                border-radius: 0 12px 12px 0; padding: 20px 24px; margin-bottom: 20px;">
        <p style="margin: 0; font-size: 15px; color: #15803d; line-height: 1.7;">
            <strong>Selamat datang!</strong> Aplikasi ini membantu mendeteksi status stunting pada balita usia
            <strong>0–60 bulan</strong> secara cepat menggunakan model <em>Machine Learning</em> yang dilatih
            dengan data dari <strong>5 Posyandu di RW 12</strong>, mengacu pada indikator
            <strong>Height-for-Age (HAZ)</strong> standar WHO.
        </p>
    </div>
    """, unsafe_allow_html=True)


    # ---- CARA PENGGUNAAN (Steps) ----
    st.markdown("### 🚀 Cara Menggunakan Aplikasi")

    step_card = """
    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px;
                padding: 20px 16px; text-align: center; min-height: 170px;
                border-top: 4px solid {color};">
        <div style="background: {color}; color: white; width: 36px; height: 36px;
                    border-radius: 50%; display: inline-flex; align-items: center;
                    justify-content: center; font-weight: 800; font-size: 16px; margin-bottom: 10px;">{num}</div>
        <p style="margin: 0 0 6px 0; font-size: 14px; font-weight: 700; color: #1e293b;">{title}</p>
        <p style="margin: 0; font-size: 12px; color: #64748b; line-height: 1.5;">{desc}</p>
    </div>
    """
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(step_card.format(color="#6366f1", num="1",
            title="Buka Prediksi", desc="Klik menu 🔍 Prediksi Stunting di sidebar kiri."), unsafe_allow_html=True)
    with c2:
        st.markdown(step_card.format(color="#0ea5e9", num="2",
            title="Isi Data Balita", desc="Masukkan Usia, Jenis Kelamin, Tinggi Badan, dan Berat Badan."), unsafe_allow_html=True)
    with c3:
        st.markdown(step_card.format(color="#f59e0b", num="3",
            title="Klik Prediksi", desc="Tekan tombol Prediksi Sekarang untuk memulai analisis."), unsafe_allow_html=True)
    with c4:
        st.markdown(step_card.format(color="#22c55e", num="4",
            title="Lihat Hasil", desc="Hasil klasifikasi, Z-Score, grafik WHO, dan rekomendasi ditampilkan."), unsafe_allow_html=True)

    st.markdown("")

    # ---- KLASIFIKASI WHO ----
    st.markdown("### 📚 Klasifikasi Status Gizi — WHO 2006")
    st.markdown("<p style='color:#64748b; font-size:14px; margin-top:-8px;'>Berdasarkan indikator Z-Score Tinggi Badan menurut Usia (TB/U)</p>", unsafe_allow_html=True)

    k1, k2, k3 = st.columns(3)

    who_card = """
    <div style="background: {bg}; border: 2px solid {border}; border-radius: 14px;
                padding: 24px 18px; text-align: center; min-height: 180px;">
        <div style="font-size: 36px; margin-bottom: 8px;">{icon}</div>
        <p style="margin: 0; font-size: 18px; font-weight: 800; color: {color};">{label}</p>
        <p style="margin: 8px 0 0 0; font-size: 13px; color: #475569; line-height: 1.6;">{criteria}</p>
        <div style="margin-top: 12px; background: {badge_bg}; padding: 4px 12px;
                    border-radius: 8px; display: inline-block;">
            <span style="font-size: 12px; font-weight: 700; color: {color};">{zscore}</span>
        </div>
    </div>
    """
    with k1:
        st.markdown(who_card.format(bg="#f0fdf4", border="#86efac", icon="✅", color="#16a34a",
            label="Normal", criteria="Tinggi badan sesuai dengan standar pertumbuhan WHO.",
            zscore="Z-Score ≥ -2 SD", badge_bg="#dcfce7"), unsafe_allow_html=True)
    with k2:
        st.markdown(who_card.format(bg="#fffbeb", border="#fcd34d", icon="⚠️", color="#d97706",
            label="Stunted", criteria="Tinggi badan di bawah standar. Perlu intervensi gizi segera.",
            zscore="-3 SD ≤ Z-Score < -2 SD", badge_bg="#fef3c7"), unsafe_allow_html=True)
    with k3:
        st.markdown(who_card.format(bg="#fef2f2", border="#fca5a5", icon="🚨", color="#dc2626",
            label="Severely Stunted", criteria="Tinggi badan jauh di bawah standar. Rujuk ke faskes segera.",
            zscore="Z-Score < -3 SD", badge_bg="#fee2e2"), unsafe_allow_html=True)

    st.markdown("")

    # ---- FITUR APLIKASI ----
    st.markdown("### ✨ Fitur Unggulan")
    f1, f2, f3 = st.columns(3)

    feature_card = """
    <div style="background: linear-gradient(135deg, {bg1}, {bg2}); border-radius: 14px;
                padding: 22px 18px; text-align: center; min-height: 150px;
                border: 1px solid {border};">
        <p style="font-size: 30px; margin: 0 0 8px 0;">{icon}</p>
        <p style="margin: 0 0 6px 0; font-size: 15px; font-weight: 700; color: {color};">{title}</p>
        <p style="margin: 0; font-size: 12px; color: #64748b; line-height: 1.5;">{desc}</p>
    </div>
    """
    with f1:
        st.markdown(feature_card.format(bg1="#eef2ff", bg2="#e0e7ff", border="#c7d2fe",
            icon="🤖", color="#4338ca", title="Prediksi AI",
            desc="Model Random Forest dengan akurasi tinggi dan optimasi GridSearchCV."), unsafe_allow_html=True)
    with f2:
        st.markdown(feature_card.format(bg1="#ecfdf5", bg2="#d1fae5", border="#a7f3d0",
            icon="📈", color="#047857", title="Grafik WHO Interaktif",
            desc="Visualisasi posisi anak pada kurva pertumbuhan standar WHO."), unsafe_allow_html=True)
    with f3:
        st.markdown(feature_card.format(bg1="#fff7ed", bg2="#ffedd5", border="#fed7aa",
            icon="📐", color="#c2410c", title="Analisis Z-Score",
            desc="Perhitungan Z-Score HAZ & WAZ lengkap dengan gauge visual SD."), unsafe_allow_html=True)

    st.markdown("")

    # ---- INFORMASI PENELITIAN ----
    st.markdown("### 🔬 Informasi Penelitian")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8fafc, #f1f5f9); border: 1px solid #e2e8f0;
                border-radius: 14px; padding: 24px 28px;">
        <p style="margin: 0 0 12px 0; font-size: 14px; color: #64748b;">Judul Skripsi:</p>
        <p style="margin: 0 0 18px 0; font-size: 15px; color: #1e293b; font-style: italic; line-height: 1.7;
                  border-left: 4px solid #6366f1; padding-left: 16px;">
            "Implementasi Machine Learning Menggunakan Algoritma Random Forest untuk Klasifikasi Dini
            Stunting pada Balita Usia 0–60 Bulan Berdasarkan Data Antropometri dengan Indikator
            Height-for-Age Standar WHO"
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    tech_items = [
        ("⚙️", "Algoritma", "Random Forest Classifier", "#6366f1"),
        ("🔄", "Optimasi", "GridSearchCV (5-Fold CV)", "#0ea5e9"),
        ("📊", "Balancing", "SMOTE Oversampling", "#f59e0b"),
        ("📍", "Data Source", "5 Posyandu — RW 12", "#22c55e"),
    ]
    tc = st.columns(4)
    for i, (icon, title, desc, color) in enumerate(tech_items):
        with tc[i]:
            st.markdown(f"""
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px;
                        padding: 16px; text-align: center; border-top: 3px solid {color};">
                <p style="margin:0; font-size: 24px;">{icon}</p>
                <p style="margin:6px 0 2px 0; font-size: 13px; font-weight: 700; color: #334155;">{title}</p>
                <p style="margin:0; font-size: 12px; color: #64748b;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ---- FAQ STUNTING ----
    st.markdown("### ❓ FAQ — Pertanyaan Umum tentang Stunting")

    with st.expander("📌 Apa itu stunting?"):
        st.markdown("""
        **Stunting** adalah kondisi gagal tumbuh pada anak balita akibat **kekurangan gizi kronis**,
        terutama pada 1.000 hari pertama kehidupan. Anak dikatakan stunting apabila **tinggi badan
        menurut usia** berada di bawah **-2 Standar Deviasi (SD)** dari median standar WHO.

        Stunting bukan hanya soal tubuh pendek — tetapi juga berdampak pada **perkembangan otak**,
        **daya tahan tubuh**, dan **produktivitas** di masa dewasa.
        """)

    with st.expander("📌 Apa penyebab utama stunting?"):
        st.markdown("""
        Penyebab stunting bersifat **multifaktor**, antara lain:
        - 🍼 **Asupan gizi tidak memadai** — kurang protein hewani, zat besi, vitamin A, zink
        - 🤒 **Infeksi berulang** — diare, ISPA, cacingan yang mengganggu penyerapan nutrisi
        - 🤰 **Gizi ibu hamil buruk** — anemia, KEK (Kurang Energi Kronik)
        - 🚰 **Sanitasi & air bersih** — lingkungan yang tidak higienis meningkatkan risiko infeksi
        - 📚 **Pola asuh** — kurangnya pengetahuan tentang pemberian MPASI yang tepat
        """)

    with st.expander("📌 Bagaimana cara mencegah stunting?"):
        st.markdown("""
        Pencegahan stunting fokus pada **1.000 Hari Pertama Kehidupan (HPK)**:
        1. ✅ **ASI Eksklusif** selama 6 bulan pertama
        2. ✅ **MPASI bergizi** mulai usia 6 bulan — tinggi protein hewani
        3. ✅ **Pemantauan rutin** di Posyandu setiap bulan
        4. ✅ **Imunisasi lengkap** sesuai jadwal
        5. ✅ **Suplementasi** — Tablet Tambah Darah untuk ibu hamil, Vitamin A untuk balita
        6. ✅ **Sanitasi** — akses air bersih dan jamban sehat
        """)

    with st.expander("📌 Bagaimana cara membaca hasil Z-Score?"):
        st.markdown("""
        **Z-Score** menunjukkan seberapa jauh tinggi badan anak dari nilai median standar WHO:

        | Z-Score | Klasifikasi | Arti |
        |:---|:---|:---|
        | ≥ -2 SD | 🟢 **Normal** | Pertumbuhan sesuai standar |
        | -3 SD s/d < -2 SD | 🟡 **Stunted** | Pendek — perlu intervensi gizi |
        | < -3 SD | 🔴 **Severely Stunted** | Sangat pendek — rujuk ke faskes |

        **Contoh:** Z-Score **-1.50 SD** berarti tinggi anak 1.5 standar deviasi di bawah median,
        namun masih tergolong **Normal** karena di atas batas -2 SD.
        """)

    st.markdown("")

    # ---- DISCLAIMER ----
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fffbeb, #fef3c7); border: 1px solid #fbbf24;
                border-radius: 12px; padding: 20px 24px; margin-top: 8px;">
        <p style="margin: 0 0 8px 0; font-size: 15px; font-weight: 700; color: #92400e;">
            ⚠️ Disclaimer Medis
        </p>
        <p style="margin: 0; font-size: 13px; color: #78350f; line-height: 1.7;">
            Aplikasi ini merupakan <strong>alat bantu deteksi dini</strong> dan <strong>bukan
            pengganti diagnosis medis profesional</strong>. Hasil prediksi harus selalu dikonfirmasi
            oleh tenaga kesehatan yang kompeten (dokter anak / petugas puskesmas). Pengembang tidak
            bertanggung jawab atas keputusan medis yang diambil hanya berdasarkan output aplikasi ini.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 12px;'>© 2025 — Sistem Deteksi Dini Stunting | Penelitian Skripsi</p>", unsafe_allow_html=True)

# ============================================================
# HALAMAN: PREDIKSI STUNTING
# ============================================================
elif menu == "🔍 Prediksi Stunting":

    # ---- BAGIAN 1: INPUT DATA (Full Width) ----
    st.markdown("<h3 style='text-align: center;'>Input Data Balita</h3>", unsafe_allow_html=True)
    
    with st.form("form_prediksi"):
        st.info("💡 Pastikan data yang dimasukkan seakurat mungkin untuk hasil terbaik.")
        
        # Slider Usia full width
        usia = st.slider("Usia (Bulan)", min_value=0, max_value=60, value=24, step=1)
        
        # 3 kolom: Jenis Kelamin | Tinggi Badan | Berat Badan
        c1, c2, c3 = st.columns(3)
        with c1:
            jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        with c2:
            tinggi_badan = st.number_input("Tinggi Badan (cm)", min_value=40.0, max_value=130.0, value=85.0, step=0.1)
        with c3:
            berat_badan = st.number_input("Berat Badan (kg)", min_value=1.0, max_value=30.0, value=11.0, step=0.1)
        
        tombol_prediksi = st.form_submit_button("🔍 Prediksi Sekarang", type="primary", use_container_width=True)

    if tombol_prediksi:
        st.session_state['prediksi_aktif'] = True

    st.markdown("---")

    # ---- BAGIAN 2: DATA BALITA (kiri) | HASIL PREDIKSI (kanan) ----
    col_data, col_hasil = st.columns(2, gap="large")

    if st.session_state.get('prediksi_aktif', False):
        # Encode & Prediksi
        jk_encoded = mapping_jk[jenis_kelamin]
        input_data = pd.DataFrame([{
            "Usia (Bulan)"     : usia,
            "Jenis Kelamin"    : jk_encoded,
            "Tinggi Badan (cm)": tinggi_badan,
            "Berat Badan (kg)" : berat_badan
        }])
        prediksi_kode  = model.predict(input_data)[0]
        prediksi_label = inverse_label[prediksi_kode]
        probabilitas   = model.predict_proba(input_data)[0]

        with col_data:
            st.markdown("#### 📋 Data Balita")
            # Baris 1: Usia | Jenis Kelamin
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("**👶 Usia**")
                st.markdown(f"""<div style="border: 2px solid #888; padding: 8px 12px; border-radius: 6px; font-size: 15px;">{usia} Bulan</div>""", unsafe_allow_html=True)
            with d2:
                st.markdown("**⚧ Gender**")
                st.markdown(f"""<div style="border: 2px solid #888; padding: 8px 12px; border-radius: 6px; font-size: 15px;">{jenis_kelamin}</div>""", unsafe_allow_html=True)
            st.markdown("")
            # Baris 2: Tinggi | Berat
            d3, d4 = st.columns(2)
            with d3:
                st.markdown("**📏 Tinggi**")
                st.markdown(f"""<div style="border: 2px solid #888; padding: 8px 12px; border-radius: 6px; font-size: 15px;">{tinggi_badan} cm</div>""", unsafe_allow_html=True)
            with d4:
                st.markdown("**⚖️ Berat**")
                st.markdown(f"""<div style="border: 2px solid #888; padding: 8px 12px; border-radius: 6px; font-size: 15px;">{berat_badan} kg</div>""", unsafe_allow_html=True)

        with col_hasil:
            st.markdown("#### 🩺 Hasil Prediksi Model")
            if prediksi_label == "Normal":
                st.success(f"**✅ {prediksi_label}**\n\nTinggi badan balita terpantau sesuai dengan standar pertumbuhan WHO.")
            elif prediksi_label == "Stunted":
                st.warning(f"**⚠️ {prediksi_label}**\n\nTinggi badan di bawah standar WHO. Disarankan pemantauan & perbaikan gizi.")
            else:
                st.error(f"**🚨 {prediksi_label}**\n\nTinggi badan jauh di bawah standar WHO. Segera rujuk ke fasilitas kesehatan.")

        st.markdown("---")

        # ---- BAGIAN 3: TINGKAT KEYAKINAN (kiri) | PENJELASAN (kanan) ----
        col_chart, col_explain = st.columns(2, gap="large")

        with col_chart:
            st.markdown("#### 📈 Tingkat Keyakinan Model")
            nama_kelas = ["Normal", "Stunted", "Severely Stunted"]
            for kelas in nama_kelas:
                prob = probabilitas[mapping_label[kelas]]
                cn, cp, cv = st.columns([3, 6, 2])
                with cn:
                    if kelas == "Normal": st.markdown("🟢 Normal")
                    elif kelas == "Stunted": st.markdown("🟡 Stunted")
                    else: st.markdown("🔴 Severely")
                with cp:
                    st.progress(float(prob))
                with cv:
                    st.markdown(f"**{prob*100:.1f}%**")

        with col_explain:
            st.markdown("#### 💡 Penjelasan Hasil")
            # Cari kelas dengan probabilitas tertinggi
            prob_normal = probabilitas[mapping_label["Normal"]]
            prob_stunted = probabilitas[mapping_label["Stunted"]]
            prob_severe = probabilitas[mapping_label["Severely Stunted"]]

            if prediksi_label == "Normal":
                st.info(
                    f"Model memprediksi status **Normal** dengan tingkat keyakinan **{prob_normal*100:.1f}%**. "
                    f"Artinya, dari keseluruhan analisis data yang dimasukkan (usia, jenis kelamin, tinggi badan, dan berat badan), "
                    f"model sangat yakin bahwa pertumbuhan balita ini **sesuai dengan standar WHO**. "
                    f"Angka keyakinan untuk Stunted hanya {prob_stunted*100:.1f}% dan Severely Stunted {prob_severe*100:.1f}%, "
                    f"menandakan risiko stunting sangat rendah."
                )
            elif prediksi_label == "Stunted":
                st.warning(
                    f"Model memprediksi status **Stunted** dengan tingkat keyakinan **{prob_stunted*100:.1f}%**. "
                    f"Ini berarti tinggi badan balita berada di bawah standar WHO untuk usianya. "
                    f"Keyakinan untuk Normal adalah {prob_normal*100:.1f}%, sehingga masih ada kemungkinan kondisi normal, "
                    f"namun **disarankan untuk segera melakukan pemeriksaan** dan meningkatkan asupan gizi, "
                    f"terutama protein hewani dan mikronutrien (Zat Besi, Vitamin A, Zink)."
                )
            else:
                st.error(
                    f"Model memprediksi status **Severely Stunted** dengan tingkat keyakinan **{prob_severe*100:.1f}%**. "
                    f"Ini menunjukkan bahwa tinggi badan balita **jauh di bawah standar WHO** untuk usianya. "
                    f"Kondisi ini memerlukan **penanganan medis segera**. Segera bawa balita ke Puskesmas atau Rumah Sakit "
                    f"untuk pemeriksaan menyeluruh dan program intervensi gizi intensif."
                )
    else:
        # Placeholder saat belum ada prediksi
        with col_data:
            st.markdown("#### 📋 Data Balita")
            st.markdown("""
            <div style="border: 2px dashed #555; border-radius: 10px; padding: 30px; text-align: center; color: #888;">
                <p>Data balita akan tampil di sini setelah Anda mengisi form dan klik Prediksi.</p>
            </div>
            """, unsafe_allow_html=True)
        with col_hasil:
            st.markdown("#### 🩺 Hasil Prediksi Model")
            st.markdown("""
            <div style="border: 2px dashed #555; border-radius: 10px; padding: 30px; text-align: center; color: #888;">
                <p>Hasil prediksi akan tampil di sini.</p>
            </div>
            """, unsafe_allow_html=True)

    # ============================================================
    # SECTION: GRAFIK PERTUMBUHAN WHO DENGAN POSISI ANAK
    # ============================================================
    if st.session_state.get('prediksi_aktif', False):
        st.markdown("---")
        st.markdown("## 📈 Posisi Anak pada Grafik Pertumbuhan WHO")
        st.markdown(f"""
        Grafik di bawah menunjukkan **posisi tinggi badan anak** (ditandai dengan titik berwarna) 
        pada kurva pertumbuhan standar WHO untuk anak **{jenis_kelamin}** usia 0–60 bulan. 
        Kurva menunjukkan garis **Median (0)**, **-2 SD**, dan **-3 SD**.
        """)
        
        try:
            chart_img = create_growth_chart_with_marker(
                jenis_kelamin, usia, tinggi_badan, prediksi_label
            )
            st.image(chart_img, use_container_width=True)
            
            # Keterangan warna titik
            if prediksi_label == "Normal":
                st.success("🟢 **Titik hijau** menunjukkan posisi anak berada di **zona normal** (di atas garis -2 SD).")
            elif prediksi_label == "Stunted":
                st.warning("🟠 **Titik oranye** menunjukkan posisi anak berada di **zona stunted** (antara garis -2 SD dan -3 SD).")
            else:
                st.error("🔴 **Titik merah** menunjukkan posisi anak berada di **zona severely stunted** (di bawah garis -3 SD).")

            st.markdown("#### 📥 Download Grafik")
            col_fmt, col_btn = st.columns([1, 3])
            with col_fmt:
                format_dl = st.selectbox("Pilih Format Unduhan:", ["PNG", "JPG", "PDF"])
            
            buf = io.BytesIO()
            if format_dl == "PNG":
                chart_img.save(buf, format="PNG")
                mime_type = "image/png"
                file_ext = "png"
            elif format_dl == "JPG":
                chart_img.convert("RGB").save(buf, format="JPEG")
                mime_type = "image/jpeg"
                file_ext = "jpg"
            else:
                chart_img.convert("RGB").save(buf, format="PDF")
                mime_type = "application/pdf"
                file_ext = "pdf"
                
            with col_btn:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                st.download_button(
                    label=f"⬇️ Download Grafik Pertumbuhan ({format_dl})",
                    data=buf.getvalue(),
                    file_name=f"Grafik_Pertumbuhan_{jenis_kelamin}_{usia}bln.{file_ext}",
                    mime=mime_type,
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Gagal memuat grafik pertumbuhan: {str(e)}")

        # ============================================================
        # SECTION: ANALISIS Z-SCORE & SD DI BAWAH GRAFIK
        # ============================================================
        h_data_chart = WHO_HEIGHT.get(jenis_kelamin, {}).get(usia)
        w_data_chart = WHO_WEIGHT.get(jenis_kelamin, {}).get(usia)

        if h_data_chart and w_data_chart:
            h_med, h_m2, h_m3 = h_data_chart
            w_med, w_m2, w_m3 = w_data_chart

            # Hitung 1 SD unit (jarak dari median ke -2SD dibagi 2)
            sd_unit_h = (h_med - h_m2) / 2
            sd_unit_w = (w_med - w_m2) / 2

            # Hitung Z-Score (Height-for-Age)
            z_score_h = (tinggi_badan - h_med) / sd_unit_h if sd_unit_h != 0 else 0
            z_score_w = (berat_badan - w_med) / sd_unit_w if sd_unit_w != 0 else 0

            # Tentukan klasifikasi berdasarkan Z-Score TB
            if z_score_h >= -2:
                zscore_status = "Normal"
                zscore_emoji = "🟢"
                zscore_color = "#22c55e"
                zscore_bg = "#f0fdf4"
                zscore_border = "#86efac"
                zscore_desc = "Tinggi badan sesuai standar WHO"
            elif z_score_h >= -3:
                zscore_status = "Stunted (Pendek)"
                zscore_emoji = "🟡"
                zscore_color = "#f59e0b"
                zscore_bg = "#fffbeb"
                zscore_border = "#fcd34d"
                zscore_desc = "Tinggi badan di bawah standar WHO"
            else:
                zscore_status = "Severely Stunted (Sangat Pendek)"
                zscore_emoji = "🔴"
                zscore_color = "#ef4444"
                zscore_bg = "#fef2f2"
                zscore_border = "#fca5a5"
                zscore_desc = "Tinggi badan jauh di bawah standar WHO"

            # Klasifikasi Z-Score BB
            if z_score_w >= -2:
                zscore_w_status = "Normal"
                zscore_w_emoji = "🟢"
                zscore_w_color = "#22c55e"
            elif z_score_w >= -3:
                zscore_w_status = "Underweight (Kurang)"
                zscore_w_emoji = "🟡"
                zscore_w_color = "#f59e0b"
            else:
                zscore_w_status = "Severely Underweight (Sangat Kurang)"
                zscore_w_emoji = "🔴"
                zscore_w_color = "#ef4444"

            st.markdown("")
            st.markdown("### 📐 Hasil Perhitungan Z-Score & Klasifikasi SD")

            # === BARIS 1: Z-Score Cards (TB dan BB) ===
            col_zh, col_zw = st.columns(2, gap="large")

            with col_zh:
                st.markdown(f"""
                <div style="background: {zscore_bg}; border: 2px solid {zscore_border}; border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="margin: 0; font-size: 13px; color: #666; text-transform: uppercase; letter-spacing: 1px;">Z-Score Tinggi Badan (HAZ)</p>
                    <p style="margin: 8px 0; font-size: 42px; font-weight: 800; color: {zscore_color};">{z_score_h:+.2f} SD</p>
                    <p style="margin: 0; font-size: 15px; color: {zscore_color}; font-weight: 600;">{zscore_emoji} {zscore_status}</p>
                    <p style="margin: 6px 0 0 0; font-size: 12px; color: #888;">{zscore_desc}</p>
                </div>
                """, unsafe_allow_html=True)

            with col_zw:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f8fafc, #f1f5f9); border: 2px solid #cbd5e1; border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="margin: 0; font-size: 13px; color: #666; text-transform: uppercase; letter-spacing: 1px;">Z-Score Berat Badan (WAZ)</p>
                    <p style="margin: 8px 0; font-size: 42px; font-weight: 800; color: {zscore_w_color};">{z_score_w:+.2f} SD</p>
                    <p style="margin: 0; font-size: 15px; color: {zscore_w_color}; font-weight: 600;">{zscore_w_emoji} {zscore_w_status}</p>
                    <p style="margin: 6px 0 0 0; font-size: 12px; color: #888;">Berat badan menurut usia</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")

            # === BARIS 2: Visual Gauge Z-Score (TB/U) ===
            st.markdown("#### 📊 Posisi Z-Score pada Skala SD (Tinggi Badan / Usia)")

            # Hitung posisi gauge (clamp antara -4 dan +4)
            gauge_z = max(-4, min(4, z_score_h))
            gauge_pct = ((gauge_z + 4) / 8) * 100  # 0-100%

            # Zona warna pada gauge
            st.markdown(f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px 24px;">
                <!-- Label zona -->
                <div style="display: flex; margin-bottom: 6px; font-size: 11px; font-weight: 600;">
                    <span style="color: #ef4444; width: 12.5%; text-align: center;">Severely Stunted</span>
                    <span style="color: #f59e0b; width: 12.5%; text-align: center;">Stunted</span>
                    <span style="color: #22c55e; width: 50%; text-align: center;">Normal</span>
                    <span style="color: #3b82f6; width: 25%; text-align: center;">Tinggi</span>
                </div>
                <!-- Gauge bar -->
                <div style="position: relative; height: 32px; border-radius: 16px; overflow: hidden; 
                            background: linear-gradient(to right, 
                                #fecaca 0%, #fecaca 12.5%, 
                                #fef08a 12.5%, #fef08a 25%, 
                                #bbf7d0 25%, #bbf7d0 75%, 
                                #bfdbfe 75%, #bfdbfe 100%);">
                    <!-- Pointer / needle -->
                    <div style="position: absolute; top: -4px; left: calc({gauge_pct:.1f}% - 12px); 
                                width: 24px; height: 40px; display: flex; flex-direction: column; align-items: center; z-index: 10;">
                        <div style="width: 4px; height: 40px; background: #1e293b; border-radius: 2px;"></div>
                    </div>
                    <!-- Dot on the bar -->
                    <div style="position: absolute; top: 50%; left: {gauge_pct:.1f}%; transform: translate(-50%, -50%); 
                                width: 20px; height: 20px; background: {zscore_color}; border: 3px solid white; 
                                border-radius: 50%; box-shadow: 0 2px 6px rgba(0,0,0,0.3); z-index: 11;"></div>
                </div>
                <!-- Scale labels -->
                <div style="display: flex; justify-content: space-between; margin-top: 6px; font-size: 12px; color: #64748b; font-weight: 500;">
                    <span>-4 SD</span>
                    <span style="position: relative; left: 2%;">-3 SD</span>
                    <span style="position: relative; left: 1%;">-2 SD</span>
                    <span>0</span>
                    <span style="position: relative; right: 1%;">+2 SD</span>
                    <span style="position: relative; right: 2%;">+3 SD</span>
                    <span>+4 SD</span>
                </div>
                <!-- Keterangan -->
                <div style="text-align: center; margin-top: 12px; padding: 8px 16px; background: white; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <span style="font-size: 14px;">Posisi anak: <strong style="color: {zscore_color}; font-size: 16px;">{z_score_h:+.2f} SD</strong></span>
                    <span style="margin-left: 12px; font-size: 13px; color: #888;">|</span>
                    <span style="margin-left: 12px; font-size: 14px;">Klasifikasi: <strong style="color: {zscore_color};">{zscore_status}</strong></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            # === BARIS 3: Detail Metrik Perbandingan TINGGI BADAN ===
            st.markdown("#### 📏 Detail Perbandingan Tinggi Badan dengan Standar WHO")

            selisih_median = tinggi_badan - h_med
            selisih_m2 = tinggi_badan - h_m2
            selisih_m3 = tinggi_badan - h_m3
            persen_median = (tinggi_badan / h_med) * 100
            tb_butuh_normal = max(0, h_m2 - tinggi_badan)

            col_m1, col_m2, col_m3 = st.columns(3)

            with col_m1:
                st.markdown(f"""
                <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; text-align: center; border-top: 4px solid #22c55e;">
                    <p style="margin: 0; font-size: 11px; color: #888; text-transform: uppercase;">Median WHO</p>
                    <p style="margin: 6px 0; font-size: 24px; font-weight: 700; color: #22c55e;">{h_med} cm</p>
                    <p style="margin: 0; font-size: 12px; color: #666;">Selisih: <b>{selisih_median:+.1f} cm</b></p>
                </div>
                """, unsafe_allow_html=True)

            with col_m2:
                st.markdown(f"""
                <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; text-align: center; border-top: 4px solid #f59e0b;">
                    <p style="margin: 0; font-size: 11px; color: #888; text-transform: uppercase;">Batas -2 SD</p>
                    <p style="margin: 6px 0; font-size: 24px; font-weight: 700; color: #f59e0b;">{h_m2} cm</p>
                    <p style="margin: 0; font-size: 12px; color: #666;">Selisih: <b>{selisih_m2:+.1f} cm</b></p>
                </div>
                """, unsafe_allow_html=True)

            with col_m3:
                st.markdown(f"""
                <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; text-align: center; border-top: 4px solid #ef4444;">
                    <p style="margin: 0; font-size: 11px; color: #888; text-transform: uppercase;">Batas -3 SD</p>
                    <p style="margin: 6px 0; font-size: 24px; font-weight: 700; color: #ef4444;">{h_m3} cm</p>
                    <p style="margin: 0; font-size: 12px; color: #666;">Selisih: <b>{selisih_m3:+.1f} cm</b></p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")

            # === BARIS 4: Detail Metrik Perbandingan BERAT BADAN ===
            st.markdown("#### ⚖️ Detail Perbandingan Berat Badan dengan Standar WHO")

            selisih_bb_med = berat_badan - w_med
            selisih_bb_m2 = berat_badan - w_m2
            selisih_bb_m3 = berat_badan - w_m3
            persen_bb_med = (berat_badan / w_med) * 100

            col_b1, col_b2, col_b3 = st.columns(3)

            with col_b1:
                st.markdown(f"""
                <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; text-align: center; border-top: 4px solid #22c55e;">
                    <p style="margin: 0; font-size: 11px; color: #888; text-transform: uppercase;">Median WHO</p>
                    <p style="margin: 6px 0; font-size: 24px; font-weight: 700; color: #22c55e;">{w_med} kg</p>
                    <p style="margin: 0; font-size: 12px; color: #666;">Selisih: <b>{selisih_bb_med:+.1f} kg</b></p>
                </div>
                """, unsafe_allow_html=True)

            with col_b2:
                st.markdown(f"""
                <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; text-align: center; border-top: 4px solid #f59e0b;">
                    <p style="margin: 0; font-size: 11px; color: #888; text-transform: uppercase;">Batas -2 SD</p>
                    <p style="margin: 6px 0; font-size: 24px; font-weight: 700; color: #f59e0b;">{w_m2} kg</p>
                    <p style="margin: 0; font-size: 12px; color: #666;">Selisih: <b>{selisih_bb_m2:+.1f} kg</b></p>
                </div>
                """, unsafe_allow_html=True)

            with col_b3:
                st.markdown(f"""
                <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; text-align: center; border-top: 4px solid #ef4444;">
                    <p style="margin: 0; font-size: 11px; color: #888; text-transform: uppercase;">Batas -3 SD</p>
                    <p style="margin: 6px 0; font-size: 24px; font-weight: 700; color: #ef4444;">{w_m3} kg</p>
                    <p style="margin: 0; font-size: 12px; color: #666;">Selisih: <b>{selisih_bb_m3:+.1f} kg</b></p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")

            # === BARIS 5: Interpretasi & Ringkasan ===
            st.markdown("#### 💡 Interpretasi & Ringkasan")

            thn = usia // 12
            bln = usia % 12
            if thn > 0 and bln > 0:
                usia_str = f"{thn} tahun {bln} bulan"
            elif thn > 0:
                usia_str = f"{thn} tahun"
            else:
                usia_str = f"{usia} bulan"

            summary_parts = []
            summary_parts.append(f"Balita **{jenis_kelamin}** berusia **{usia_str}** dengan tinggi badan **{tinggi_badan} cm** dan berat badan **{berat_badan} kg**.")
            summary_parts.append("")

            # Analisis tinggi badan
            if tinggi_badan >= h_med:
                summary_parts.append(f"📏 **Tinggi Badan:** Mencapai atau melebihi median WHO ({h_med} cm). Pertumbuhan tinggi badan **sangat baik**.")
            elif tinggi_badan >= h_m2:
                summary_parts.append(f"📏 **Tinggi Badan:** Masih dalam rentang normal WHO, namun **{abs(selisih_median):.1f} cm di bawah** median ({h_med} cm). Pertumbuhan perlu terus dioptimalkan dengan asupan gizi yang baik.")
            elif tinggi_badan >= h_m3:
                summary_parts.append(f"📏 **Tinggi Badan:** Berada di bawah batas -2 SD ({h_m2} cm), yaitu **{abs(selisih_median):.1f} cm di bawah** median WHO. Anak terindikasi **stunted** dan memerlukan peningkatan asupan gizi serta pemantauan pertumbuhan secara rutin.")
            else:
                summary_parts.append(f"📏 **Tinggi Badan:** Berada di bawah batas -3 SD ({h_m3} cm), yaitu **{abs(selisih_median):.1f} cm di bawah** median WHO. Anak terindikasi **severely stunted** dan memerlukan penanganan medis segera.")

            summary_parts.append("")

            # Analisis berat badan
            if berat_badan >= w_med:
                summary_parts.append(f"⚖️ **Berat Badan:** Mencapai atau melebihi median WHO ({w_med} kg). Berat badan **sangat baik**.")
            elif berat_badan >= w_m2:
                summary_parts.append(f"⚖️ **Berat Badan:** Masih dalam rentang normal, namun **{abs(selisih_bb_med):.1f} kg di bawah** median WHO ({w_med} kg). Perlu dipantau agar tidak terus menurun, perbanyak asupan protein dan kalori.")
            elif berat_badan >= w_m3:
                summary_parts.append(f"⚖️ **Berat Badan:** Di bawah batas -2 SD ({w_m2} kg). Anak terindikasi **underweight** (kurus) dan memerlukan peningkatan asupan gizi, terutama protein hewani dan lemak sehat.")
            else:
                summary_parts.append(f"⚖️ **Berat Badan:** Di bawah batas -3 SD ({w_m3} kg). Anak terindikasi **severely underweight** (sangat kurus) dan memerlukan penanganan medis segera serta program gizi intensif.")

            st.info("\n".join(summary_parts))

            # --- Rekomendasi ---
            st.markdown("---")
            st.markdown("### 🩺 Rekomendasi Tindak Lanjut")

            if prediksi_label == "Normal":
                st.success("""
                ✅ **Status pertumbuhan anak dalam batas normal.** Untuk menjaga agar tetap optimal:
                - Berikan **ASI eksklusif** hingga usia 6 bulan dan lanjutkan hingga 2 tahun
                - Berikan **MPASI bergizi seimbang** sesuai usia (karbohidrat, protein hewani, sayur, buah)
                - Lakukan **pemantauan rutin** di Posyandu setiap bulan
                - Pastikan **imunisasi lengkap** sesuai jadwal
                - Jaga **kebersihan lingkungan** dan sanitasi untuk mencegah infeksi berulang
                """)
            elif prediksi_label == "Stunted":
                st.warning(f"""
                ⚠️ **Anak terindikasi stunted.** Tinggi badan berada {abs(selisih_median):.1f} cm di bawah median WHO. Langkah yang disarankan:
                - **Segera konsultasi** ke dokter anak atau petugas gizi Puskesmas
                - Tingkatkan asupan **protein hewani** (telur, ikan, daging, susu)
                - Berikan **makanan padat gizi** dengan frekuensi lebih sering
                - Pastikan anak mendapatkan **vitamin dan mineral** yang cukup (Zink, Vitamin A, Zat Besi)
                - Lakukan **pemeriksaan kesehatan rutin** untuk deteksi penyakit penyerta
                - Pantau pertumbuhan **setiap 2 minggu** di Posyandu
                """)
            else:
                st.error(f"""
                🚨 **Anak terindikasi severely stunted.** Tinggi badan berada {abs(selisih_median):.1f} cm di bawah median WHO. Tindakan yang harus segera dilakukan:
                - **Segera bawa ke fasilitas kesehatan** (Puskesmas/Rumah Sakit) untuk pemeriksaan menyeluruh
                - Anak kemungkinan memerlukan **program Pemberian Makanan Tambahan (PMT)** dari pemerintah
                - Dokter mungkin akan memberikan **suplementasi gizi khusus**
                - Periksa kemungkinan **penyakit infeksi kronis** (diare berulang, TBC, cacingan)
                - Evaluasi **pola makan dan pola asuh** bersama tenaga kesehatan
                - Pemantauan pertumbuhan harus dilakukan **lebih intensif** (setiap minggu)
                """)

            st.caption("📌 *Analisis ini berdasarkan WHO Child Growth Standards 2006 dan bersifat informatif. Selalu konsultasikan dengan tenaga kesehatan profesional.*")

        else:
            st.warning(f"Data standar WHO untuk usia {usia} bulan tidak tersedia dalam database.")

