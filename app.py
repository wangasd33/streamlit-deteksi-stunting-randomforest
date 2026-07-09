import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import base64
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io
import copy
import math

BASE_DIR = Path(__file__).resolve().parent

def find_file(filename, subfolders=None):
    """
    Mencari file secara dinamis di direktori saat ini atau subdirektori terkait.
    Memastikan aplikasi berjalan sempurna baik di root folder maupun di dalam folder modelml_rf.
    """
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "modelml_rf" / filename,
        BASE_DIR / "models" / filename,
        BASE_DIR / "modelml_rf" / "models" / filename,
    ]
    if subfolders:
        for sf in subfolders:
            candidates.append(BASE_DIR / sf / filename)
            candidates.append(BASE_DIR / "modelml_rf" / sf / filename)
            candidates.append(BASE_DIR / "assets" / "images" / filename)
            candidates.append(BASE_DIR / "modelml_rf" / "assets" / "images" / filename)
    for path in candidates:
        if path.exists():
            return path
    return None

def get_base64_image(image_path):
    try:
        if image_path and image_path.exists():
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception:
        pass
    return ""

chart_laki = find_file("Grafik anak laki-laki - Panjang atau tinggi menurut usia dari lahir hingga 5 tahun (skor z)_page-0001.jpg", ["assets/images"])
chart_perempuan = find_file("Grafik anak perempuan - Panjang atau tinggi badan menurut usia dari lahir hingga 5 tahun (skor z)_page-0001.jpg", ["assets/images"])

# ============================================================
# FUNGSI OVERLAY TITIK PADA GRAFIK WHO
# ============================================================
CHART_CALIBRATION = {
    "Laki-laki": {
        "image": chart_laki,
        "birth_x": 461.0,    # pixel x tepat di garis usia 0 bulan (Birth)
        "ppm": 42.0,         # pixels per month (24 bulan = x=1469 tepat di garis 2 years)
        "y_45": 2014.0,      # pixel y terkalibrasi tepat dengan kurva WHO SD0 dan SD-2
        "ppc": 18.96,        # pixels per cm (85 cm = y=1255 tepat di z-score -0.69 SD)
    },
    "Perempuan": {
        "image": chart_perempuan,
        "birth_x": 461.0,
        "ppm": 42.0,
        "y_45": 2014.0,
        "ppc": 18.96,
    }
}

def get_chart_font(size=38, bold=False):
    """
    Memuat font TrueType berukuran proporsional (sedang) agar jelas terbaca dan rapi.
    """
    if bold:
        candidates = [
            "arialbd.ttf", "Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
        ]
    else:
        candidates = [
            "arial.ttf", "Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]
    for fn in candidates:
        try:
            return ImageFont.truetype(fn, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

def create_growth_chart_with_marker(jenis_kelamin, usia_bulan, tinggi_cm, prediksi_label):
    """
    Membuat gambar grafik WHO dengan overlay titik posisi anak berukuran sedang proporsional & tepat sasaran.
    """
    cal = CHART_CALIBRATION.get(jenis_kelamin)
    if not cal or not cal["image"] or not cal["image"].exists():
        return None
    
    img = Image.open(cal["image"]).convert("RGBA")
    
    px = cal["birth_x"] + usia_bulan * cal["ppm"]
    py = cal["y_45"] - (tinggi_cm - 45) * cal["ppc"]
    
    px = max(0, min(px, img.width - 1))
    py = max(0, min(py, img.height - 1))
    
    if prediksi_label == "Normal":
        dot_color = (34, 197, 94, 255)
        outline_color = (21, 128, 61, 255)
        label_text = "Normal"
    elif prediksi_label == "Stunted":
        dot_color = (251, 146, 60, 255)
        outline_color = (194, 65, 12, 255)
        label_text = "Stunted"
    else:
        dot_color = (239, 68, 68, 255)
        outline_color = (185, 28, 28, 255)
        label_text = "Severely Stunted"
    
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    dot_radius = 18
    outline_width = 4
    
    draw.ellipse(
        [px - dot_radius - outline_width, py - dot_radius - outline_width,
         px + dot_radius + outline_width, py + dot_radius + outline_width],
        fill=outline_color
    )
    
    draw.ellipse(
        [px - dot_radius, py - dot_radius, px + dot_radius, py + dot_radius],
        fill=dot_color
    )
    
    highlight_r = 5
    draw.ellipse(
        [px - highlight_r, py - highlight_r,
         px + highlight_r, py + highlight_r],
        fill=(255, 255, 255, 210)
    )
    
    font_bold = get_chart_font(size=38, bold=True)
    font_reg = get_chart_font(size=32, bold=False)
    
    label_info = f"{tinggi_cm} cm, {usia_bulan} bln"
    
    bbox1 = draw.textbbox((0, 0), label_text, font=font_bold)
    bbox2 = draw.textbbox((0, 0), label_info, font=font_reg)
    
    w1 = bbox1[2] - bbox1[0]
    h1 = bbox1[3] - bbox1[1]
    w2 = bbox2[2] - bbox2[0]
    h2 = bbox2[3] - bbox2[1]
    
    pad_x = 18
    pad_y = 12
    text_w = max(w1, w2) + (pad_x * 2)
    text_h = h1 + h2 + (pad_y * 2) + 8
    
    label_x = px + dot_radius + 15
    label_y = py - dot_radius - 22
    
    if label_x + text_w > img.width - 20:
        label_x = px - dot_radius - text_w - 15
    if label_y < 20:
        label_y = py + dot_radius + 15
    
    bg_x1 = label_x
    bg_y1 = label_y
    bg_x2 = bg_x1 + text_w
    bg_y2 = bg_y1 + text_h
    
    draw.rounded_rectangle(
        [bg_x1, bg_y1, bg_x2, bg_y2],
        radius=8,
        fill=(255, 255, 255, 245),
        outline=outline_color,
        width=2
    )
    
    draw.text((label_x + pad_x, label_y + pad_y - 2), label_text, fill=outline_color, font=font_bold)
    draw.text((label_x + pad_x, label_y + pad_y + h1 + 6), label_info, fill=(30, 41, 59, 255), font=font_reg)
    
    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB")

icon_path = find_file("icon-balita.jpg", ["assets/images"])
logo_path = find_file("logo-balita.jpg", ["assets/images"])

icon_base64 = get_base64_image(icon_path)
icon_html = f"<img src='data:image/jpeg;base64,{icon_base64}' width='45' style='vertical-align: middle; margin-right: 10px; margin-bottom: 5px; border-radius: 50%;'>" if icon_base64 else "👶 "

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Deteksi Dini Stunting",
    page_icon=Image.open(icon_path) if icon_path and icon_path.exists() else "👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD MODEL & MAPPING
# ============================================================
@st.cache_resource
def load_model():
    model_path = find_file("model_rf_stunting_terbaik.pkl")
    if model_path is None:
        return None, None, None, None, None
    model = joblib.load(model_path)
    
    mapping_label = {"Normal": 0, "Stunted": 1, "Severely Stunted": 2}
    mapping_jk    = {"Laki-laki": 0, "Perempuan": 1}
    inverse_label = {v: k for k, v in mapping_label.items()}
    inverse_jk    = {v: k for k, v in mapping_jk.items()}
    return model, mapping_label, mapping_jk, inverse_label, inverse_jk

model, mapping_label, mapping_jk, inverse_label, inverse_jk = load_model()

# ============================================================
# DATA STANDAR WHO (Height-for-Age & Weight-for-Age)
# Sumber: WHO Child Growth Standards 2006
#
# Tabel di bawah menyimpan parameter LMS asli dari WHO, BUKAN nilai
# ambang (-2 SD / -3 SD) yang diketik manual. Seluruh ambang batas
# dihitung dari parameter ini, sehingga tidak dapat menyimpang dari
# notebook maupun dari tabel resmi WHO.
#
# Validasi:
#   - TB/U : M - 2*SD dan M - 3*SD cocok dengan kolom -2SD/-3SD PDF
#            WHO pada seluruh 122 baris.
#   - BB/U : rumus LMS mereproduksi 366 titik ambang (-3/-2/-1 SD dan
#            median) pada PDF WHO tanpa deviasi.
# ============================================================

# ------------------------------------------------------------
# TABEL 1 - Length/Height-for-age (TB/U)
# Usia 0-24 bulan  : tabel Length-for-age (diukur berbaring)
# Usia 25-60 bulan : tabel Height-for-age (diukur berdiri)
# Format: usia_bulan: (M, SD)
#
# Kolom L pada tabel TB/U bernilai 1 di SELURUH baris, sehingga
# rumus LMS  Z = ((X/M)^L - 1) / (L*S)  tereduksi secara aljabar
# menjadi  Z = (X - M) / SD. Keduanya ekuivalen, bukan penyederhanaan.
# ------------------------------------------------------------
WHO_HAZ = {
    "Laki-laki": {
        0: (49.8842, 1.8931), 1: (54.7244, 1.9465), 2: (58.4249, 2.0005),
        3: (61.4292, 2.0444), 4: (63.8860, 2.0808), 5: (65.9026, 2.1115),
        6: (67.6236, 2.1403), 7: (69.1645, 2.1711), 8: (70.5994, 2.2055),
        9: (71.9687, 2.2433), 10: (73.2812, 2.2849), 11: (74.5388, 2.3293),
        12: (75.7488, 2.3762), 13: (76.9186, 2.4260), 14: (78.0497, 2.4773),
        15: (79.1458, 2.5303), 16: (80.2113, 2.5844), 17: (81.2487, 2.6406),
        18: (82.2587, 2.6973), 19: (83.2418, 2.7553), 20: (84.1996, 2.8140),
        21: (85.1348, 2.8742), 22: (86.0477, 2.9342), 23: (86.9410, 2.9951),
        24: (87.8161, 3.0551), 25: (87.9720, 3.1160), 26: (88.8065, 3.1757),
        27: (89.6197, 3.2353), 28: (90.4120, 3.2928), 29: (91.1828, 3.3501),
        30: (91.9327, 3.4052), 31: (92.6631, 3.4591), 32: (93.3753, 3.5118),
        33: (94.0711, 3.5625), 34: (94.7532, 3.6120), 35: (95.4236, 3.6604),
        36: (96.0835, 3.7069), 37: (96.7337, 3.7523), 38: (97.3749, 3.7976),
        39: (98.0073, 3.8409), 40: (98.6310, 3.8831), 41: (99.2459, 3.9242),
        42: (99.8515, 3.9651), 43: (100.4485, 4.0039), 44: (101.0374, 4.0435),
        45: (101.6186, 4.0810), 46: (102.1933, 4.1194), 47: (102.7625, 4.1567),
        48: (103.3273, 4.1941), 49: (103.8886, 4.2314), 50: (104.4473, 4.2677),
        51: (105.0041, 4.3052), 52: (105.5596, 4.3417), 53: (106.1138, 4.3783),
        54: (106.6668, 4.4149), 55: (107.2188, 4.4517), 56: (107.7697, 4.4886),
        57: (108.3198, 4.5245), 58: (108.8689, 4.5616), 59: (109.4170, 4.5977),
        60: (109.9638, 4.6339),
    },
    "Perempuan": {
        0: (49.1477, 1.8627), 1: (53.6872, 1.9542), 2: (57.0673, 2.0362),
        3: (59.8029, 2.1051), 4: (62.0899, 2.1645), 5: (64.0301, 2.2174),
        6: (65.7311, 2.2664), 7: (67.2873, 2.3154), 8: (68.7498, 2.3650),
        9: (70.1435, 2.4157), 10: (71.4818, 2.4676), 11: (72.7710, 2.5208),
        12: (74.0150, 2.5750), 13: (75.2176, 2.6296), 14: (76.3817, 2.6841),
        15: (77.5099, 2.7392), 16: (78.6055, 2.7944), 17: (79.6710, 2.8490),
        18: (80.7079, 2.9039), 19: (81.7182, 2.9582), 20: (82.7036, 3.0129),
        21: (83.6654, 3.0672), 22: (84.6040, 3.1202), 23: (85.5202, 3.1737),
        24: (86.4153, 3.2267), 25: (86.5904, 3.2783), 26: (87.4462, 3.3300),
        27: (88.2830, 3.3812), 28: (89.1004, 3.4313), 29: (89.8991, 3.4809),
        30: (90.6797, 3.5302), 31: (91.4430, 3.5782), 32: (92.1906, 3.6259),
        33: (92.9239, 3.6724), 34: (93.6444, 3.7186), 35: (94.3533, 3.7638),
        36: (95.0515, 3.8078), 37: (95.7399, 3.8526), 38: (96.4187, 3.8963),
        39: (97.0885, 3.9389), 40: (97.7493, 3.9813), 41: (98.4015, 4.0236),
        42: (99.0448, 4.0658), 43: (99.6795, 4.1068), 44: (100.3058, 4.1476),
        45: (100.9238, 4.1883), 46: (101.5337, 4.2279), 47: (102.1360, 4.2683),
        48: (102.7312, 4.3075), 49: (103.3197, 4.3456), 50: (103.9021, 4.3847),
        51: (104.4786, 4.4226), 52: (105.0494, 4.4604), 53: (105.6148, 4.4981),
        54: (106.1748, 4.5358), 55: (106.7295, 4.5734), 56: (107.2788, 4.6108),
        57: (107.8227, 4.6472), 58: (108.3613, 4.6834), 59: (108.8948, 4.7195),
        60: (109.4233, 4.7566),
    },
}

# ------------------------------------------------------------
# TABEL 2 - Weight-for-age (BB/U)
# Format: usia_bulan: (L, M, S)
#
# PENTING: L != 1 pada tabel BB/U (laki-laki: +0.3487 -> -0.1506).
# Distribusi berat badan MIRING, sehingga rumus simetris
# (BB - M)/SD TIDAK SAH. Wajib memakai rumus LMS penuh.
# ------------------------------------------------------------
WHO_WFA = {
    "Laki-laki": {
        0: (0.3487, 3.3464, 0.14602), 1: (0.2297, 4.4709, 0.13395),
        2: (0.1970, 5.5675, 0.12385), 3: (0.1738, 6.3762, 0.11727),
        4: (0.1553, 7.0023, 0.11316), 5: (0.1395, 7.5105, 0.11080),
        6: (0.1257, 7.9340, 0.10958), 7: (0.1134, 8.2970, 0.10902),
        8: (0.1021, 8.6151, 0.10882), 9: (0.0917, 8.9014, 0.10881),
        10: (0.0820, 9.1649, 0.10891), 11: (0.0730, 9.4122, 0.10906),
        12: (0.0644, 9.6479, 0.10925), 13: (0.0563, 9.8749, 0.10949),
        14: (0.0487, 10.0953, 0.10976), 15: (0.0413, 10.3108, 0.11007),
        16: (0.0343, 10.5228, 0.11041), 17: (0.0275, 10.7319, 0.11079),
        18: (0.0211, 10.9385, 0.11119), 19: (0.0148, 11.1430, 0.11164),
        20: (0.0087, 11.3462, 0.11211), 21: (0.0029, 11.5486, 0.11261),
        22: (-0.0028, 11.7504, 0.11314), 23: (-0.0083, 11.9514, 0.11369),
        24: (-0.0137, 12.1515, 0.11426), 25: (-0.0189, 12.3502, 0.11485),
        26: (-0.0240, 12.5466, 0.11544), 27: (-0.0289, 12.7401, 0.11604),
        28: (-0.0337, 12.9303, 0.11664), 29: (-0.0385, 13.1169, 0.11723),
        30: (-0.0431, 13.3000, 0.11781), 31: (-0.0476, 13.4798, 0.11839),
        32: (-0.0520, 13.6567, 0.11896), 33: (-0.0564, 13.8309, 0.11953),
        34: (-0.0606, 14.0031, 0.12008), 35: (-0.0648, 14.1736, 0.12062),
        36: (-0.0689, 14.3429, 0.12116), 37: (-0.0729, 14.5113, 0.12168),
        38: (-0.0769, 14.6791, 0.12220), 39: (-0.0808, 14.8466, 0.12271),
        40: (-0.0846, 15.0140, 0.12322), 41: (-0.0883, 15.1813, 0.12373),
        42: (-0.0920, 15.3486, 0.12425), 43: (-0.0957, 15.5158, 0.12478),
        44: (-0.0993, 15.6828, 0.12531), 45: (-0.1028, 15.8497, 0.12586),
        46: (-0.1063, 16.0163, 0.12643), 47: (-0.1097, 16.1827, 0.12700),
        48: (-0.1131, 16.3489, 0.12759), 49: (-0.1165, 16.5150, 0.12819),
        50: (-0.1198, 16.6811, 0.12880), 51: (-0.1230, 16.8471, 0.12943),
        52: (-0.1262, 17.0132, 0.13005), 53: (-0.1294, 17.1792, 0.13069),
        54: (-0.1325, 17.3452, 0.13133), 55: (-0.1356, 17.5111, 0.13197),
        56: (-0.1387, 17.6768, 0.13261), 57: (-0.1417, 17.8422, 0.13325),
        58: (-0.1447, 18.0073, 0.13389), 59: (-0.1477, 18.1722, 0.13453),
        60: (-0.1506, 18.3366, 0.13517),
    },
    "Perempuan": {
        0: (0.3809, 3.2322, 0.14171), 1: (0.1714, 4.1873, 0.13724),
        2: (0.0962, 5.1282, 0.13000), 3: (0.0402, 5.8458, 0.12619),
        4: (-0.0050, 6.4237, 0.12402), 5: (-0.0430, 6.8985, 0.12274),
        6: (-0.0756, 7.2970, 0.12204), 7: (-0.1039, 7.6422, 0.12178),
        8: (-0.1288, 7.9487, 0.12181), 9: (-0.1507, 8.2254, 0.12199),
        10: (-0.1700, 8.4800, 0.12223), 11: (-0.1872, 8.7192, 0.12247),
        12: (-0.2024, 8.9481, 0.12268), 13: (-0.2158, 9.1699, 0.12283),
        14: (-0.2278, 9.3870, 0.12294), 15: (-0.2384, 9.6008, 0.12299),
        16: (-0.2478, 9.8124, 0.12303), 17: (-0.2562, 10.0226, 0.12306),
        18: (-0.2637, 10.2315, 0.12309), 19: (-0.2703, 10.4393, 0.12315),
        20: (-0.2762, 10.6464, 0.12323), 21: (-0.2815, 10.8534, 0.12335),
        22: (-0.2862, 11.0608, 0.12350), 23: (-0.2903, 11.2688, 0.12369),
        24: (-0.2941, 11.4775, 0.12390), 25: (-0.2975, 11.6864, 0.12414),
        26: (-0.3005, 11.8947, 0.12441), 27: (-0.3032, 12.1015, 0.12472),
        28: (-0.3057, 12.3059, 0.12506), 29: (-0.3080, 12.5073, 0.12545),
        30: (-0.3101, 12.7055, 0.12587), 31: (-0.3120, 12.9006, 0.12633),
        32: (-0.3138, 13.0930, 0.12683), 33: (-0.3155, 13.2837, 0.12737),
        34: (-0.3171, 13.4731, 0.12794), 35: (-0.3186, 13.6618, 0.12855),
        36: (-0.3201, 13.8503, 0.12919), 37: (-0.3216, 14.0385, 0.12988),
        38: (-0.3230, 14.2265, 0.13059), 39: (-0.3243, 14.4140, 0.13135),
        40: (-0.3257, 14.6010, 0.13213), 41: (-0.3270, 14.7873, 0.13293),
        42: (-0.3283, 14.9727, 0.13376), 43: (-0.3296, 15.1573, 0.13460),
        44: (-0.3309, 15.3410, 0.13545), 45: (-0.3322, 15.5240, 0.13630),
        46: (-0.3335, 15.7064, 0.13716), 47: (-0.3348, 15.8882, 0.13800),
        48: (-0.3361, 16.0697, 0.13884), 49: (-0.3374, 16.2511, 0.13968),
        50: (-0.3387, 16.4322, 0.14051), 51: (-0.3400, 16.6133, 0.14132),
        52: (-0.3414, 16.7942, 0.14213), 53: (-0.3427, 16.9748, 0.14293),
        54: (-0.3440, 17.1551, 0.14371), 55: (-0.3453, 17.3347, 0.14448),
        56: (-0.3466, 17.5136, 0.14525), 57: (-0.3479, 17.6916, 0.14600),
        58: (-0.3492, 17.8686, 0.14675), 59: (-0.3505, 18.0445, 0.14748),
        60: (-0.3518, 18.2193, 0.14821),
    },
}


# ============================================================
# FUNGSI PERHITUNGAN Z-SCORE (mengacu WHO Child Growth Standards)
# ============================================================
def hitung_haz(tinggi_cm, M, SD):
    """
    Z-Score Tinggi Badan menurut Usia (TB/U).

    Rumus LMS WHO: Z = ((X/M)^L - 1) / (L*S)
    Karena L = 1 pada seluruh tabel height-for-age, rumus tersebut
    tereduksi menjadi Z = (X - M) / SD.

    Tidak ada koreksi nilai ekstrem: distribusi TB/U sudah normal.
    """
    return (tinggi_cm - M) / SD


def nilai_pada_z(z, L, M, S):
    """Mengembalikan nilai pengukuran pada Z-Score tertentu (invers LMS)."""
    if abs(L) < 1e-9:
        return M * math.exp(S * z)
    return M * pow(1 + L * S * z, 1 / L)


def hitung_waz(berat_kg, L, M, S):
    """
    Z-Score Berat Badan menurut Usia (BB/U) - rumus LMS penuh WHO.

        Z = ((BB/M)^L - 1) / (L*S)

    Untuk |Z| > 3, WHO mensyaratkan koreksi linier karena ekor
    distribusi LMS tidak realistis di luar +/-3 SD. Koreksi ini
    berlaku bagi indikator berbasis berat badan (BB/U, BB/TB, IMT/U)
    dan TIDAK berlaku bagi TB/U.
    """
    if abs(L) < 1e-9:
        z = math.log(berat_kg / M) / S
    else:
        z = ((berat_kg / M) ** L - 1) / (L * S)

    if z > 3:
        sd3pos = nilai_pada_z(3, L, M, S)
        sd2pos = nilai_pada_z(2, L, M, S)
        z = 3 + (berat_kg - sd3pos) / (sd3pos - sd2pos)
    elif z < -3:
        sd3neg = nilai_pada_z(-3, L, M, S)
        sd2neg = nilai_pada_z(-2, L, M, S)
        z = -3 + (berat_kg - sd3neg) / (sd2neg - sd3neg)

    return z


# ============================================================
# SIDEBAR NAVIGASI
# ============================================================
with st.sidebar:
    if logo_path and logo_path.exists():
        st.image(str(logo_path), width=100)
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

    st.markdown("### 🔬 Informasi Penelitian")

    st.markdown("""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px; padding: 24px; margin-bottom: 20px;">
        <p style="margin: 0 0 10px 0; font-size: 13px; color: #64748b; font-weight: 600;">Judul Skripsi:</p>
        <div style="border-left: 4px solid #6366f1; padding-left: 16px; margin: 4px 0;">
            <p style="margin: 0; font-size: 15.5px; font-weight: 700; color: #1e293b; font-style: italic; line-height: 1.6;">
                "Implementasi Machine Learning Menggunakan Algoritma Random Forest untuk Klasifikasi Dini Stunting pada Balita Usia 0–60 Bulan Berdasarkan Data Antropometri dengan Indikator Height-for-Age Standar WHO"
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    info_card = """
    <div style="background: white; border: 1px solid #e2e8f0; border-top: 4px solid {border};
                border-radius: 12px; padding: 18px 14px; text-align: center; min-height: 125px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
        <div style="font-size: 26px; margin-bottom: 8px;">{icon}</div>
        <p style="margin: 0 0 6px 0; font-size: 14px; font-weight: 700; color: #1e293b;">{title}</p>
        <p style="margin: 0; font-size: 12.5px; color: #64748b; line-height: 1.4;">{value}</p>
    </div>
    """

    i1, i2, i3, i4 = st.columns(4)
    with i1:
        st.markdown(info_card.format(border="#6366f1", icon="⚙️", title="Algoritma", value="Random Forest Classifier"), unsafe_allow_html=True)
    with i2:
        st.markdown(info_card.format(border="#0ea5e9", icon="🔄", title="Optimasi", value="GridSearchCV (5-Fold CV)"), unsafe_allow_html=True)
    with i3:
        st.markdown(info_card.format(border="#f59e0b", icon="📊", title="Balancing", value="SMOTE Oversampling"), unsafe_allow_html=True)
    with i4:
        st.markdown(info_card.format(border="#22c55e", icon="📍", title="Data Source", value="5 Posyandu — RW 12"), unsafe_allow_html=True)

    st.markdown("")

    j1, j2 = st.columns(2)
    with j1:
        st.markdown("""
        <div style="background: white; border: 1px solid #e2e8f0; border-top: 4px solid #8b5cf6;
                    border-radius: 12px; padding: 18px 18px; text-align: center; min-height: 115px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <div style="font-size: 26px; margin-bottom: 6px;">🎯</div>
            <p style="margin: 0 0 6px 0; font-size: 14px; font-weight: 700; color: #1e293b;">Indikator Target</p>
            <p style="margin: 0; font-size: 13px; color: #64748b; line-height: 1.5;">
                Height-for-Age Z-Score (HAZ / TB/U) mengacu standar WHO 2006
            </p>
        </div>
        """, unsafe_allow_html=True)

    with j2:
        st.markdown("""
        <div style="background: white; border: 1px solid #e2e8f0; border-top: 4px solid #ec4899;
                    border-radius: 12px; padding: 18px 18px; text-align: center; min-height: 115px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <div style="font-size: 26px; margin-bottom: 6px;">🏷️</div>
            <p style="margin: 0 0 10px 0; font-size: 14px; font-weight: 700; color: #1e293b;">Kelas Output</p>
            <div>
                <span style="background: #dcfce7; color: #16a34a; font-size: 12px; font-weight: 700; padding: 4px 10px; border-radius: 6px; margin: 0 3px;">Normal</span>
                <span style="background: #fef3c7; color: #d97706; font-size: 12px; font-weight: 700; padding: 4px 10px; border-radius: 6px; margin: 0 3px;">Stunted</span>
                <span style="background: #fee2e2; color: #dc2626; font-size: 12px; font-weight: 700; padding: 4px 10px; border-radius: 6px; margin: 0 3px;">Severely Stunted</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("### ❓ FAQ — Pertanyaan Umum tentang Stunting")

    with st.expander("📌 Apa itu stunting?"):
        st.markdown("""
**Stunting** adalah kondisi gagal tumbuh pada anak balita akibat **kekurangan gizi kronis**, terutama pada 1.000 hari pertama kehidupan. Anak dikatakan stunting apabila **tinggi badan menurut usia** berada di bawah **-2 Standar Deviasi (SD)** dari median standar WHO.

Stunting bukan hanya soal tubuh pendek — tetapi juga berdampak pada **perkembangan otak, daya tahan tubuh,** dan **produktivitas** di masa dewasa.
        """)

    with st.expander("📌 Apa penyebab utama stunting?"):
        st.markdown("""
Penyebab stunting bersifat **multifaktor**, antara lain:

- 🍼 **Asupan gizi tidak memadai** — kurang protein hewani, zat besi, vitamin A, zink
- 🤒 **Infeksi berulang** — diare, ISPA, cacingan yang mengganggu penyerapan nutrisi
- 🤰 **Gizi ibu hamil buruk** — anemia, KEK (Kurang Energi Kronik)
- 🚰 **Sanitasi & air bersih** — lingkungan yang tidak higienis meningkatkan risiko infeksi
- 🧸 **Pola asuh** — kurangnya pengetahuan tentang pemberian MPASI yang tepat
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
| :--- | :--- | :--- |
| ≥ -2 SD | 🟢 **Normal** | Pertumbuhan sesuai standar |
| -3 SD s/d < -2 SD | 🟡 **Stunted** | Pendek — perlu intervensi gizi |
| < -3 SD | 🔴 **Severely Stunted** | Sangat pendek — rujuk ke faskes |

**Contoh:** Z-Score **-1.50 SD** berarti tinggi anak 1.5 standar deviasi di bawah median, namun masih tergolong **Normal** karena di atas batas -2 SD.
        """)

# ============================================================
# HALAMAN: PREDIKSI STUNTING
# ============================================================
elif menu == "🔍 Prediksi Stunting":

    st.markdown("## 🔍 Prediksi Status Gizi Balita")
    st.markdown("Masukkan data antropometri balita untuk melakukan klasifikasi dini status stunting.")
    st.markdown("---")

    if model is None:
        st.error("⚠️ **Model Machine Learning tidak ditemukan!** Pastikan file `model_rf_stunting_terbaik.pkl` tersedia di direktori aplikasi atau folder `modelml_rf`.")
    else:
        # ---- BAGIAN 1: INPUT DATA ----
        with st.form("form_prediksi"):
            st.markdown("#### 📋 Form Antropometri Balita")
            st.markdown("""
            <div style="background-color: #fffbeb; border: 1px solid #fcd34d; border-radius: 8px; padding: 12px 16px; margin: 8px 0 16px 0;">
                <p style="margin: 0; color: #92400e; font-size: 14px; font-weight: 500;">
                    💡 Pastikan data yang dimasukkan seakurat mungkin untuk hasil terbaik.
                </p>
            </div>
            <style>
            div[data-testid="stFormSubmitButton"] > button {
                background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
                color: #ffffff !important;
                border: none !important;
                font-weight: 700 !important;
                font-size: 15px !important;
                border-radius: 8px !important;
                padding: 10px 24px !important;
                box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
                transition: all 0.2s ease !important;
            }
            div[data-testid="stFormSubmitButton"] > button:hover {
                background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
                color: #ffffff !important;
                box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4) !important;
            }
            </style>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                usia = st.number_input("Usia (Bulan):", min_value=0, max_value=60, value=24, step=1,
                                       help="Usia balita antara 0 - 60 bulan")
                tinggi_badan = st.number_input("Tinggi Badan / Panjang Badan (cm):", min_value=30.0, max_value=130.0,
                                               value=85.0, step=0.1, help="Tinggi badan saat ini")
            with col2:
                jenis_kelamin = st.selectbox("Jenis Kelamin:", ["Laki-laki", "Perempuan"])
                berat_badan = st.number_input("Berat Badan (kg):", min_value=2.0, max_value=35.0,
                                              value=12.0, step=0.1, help="Berat badan saat ini")

            tombol_prediksi = st.form_submit_button("🔍 Prediksi Sekarang", type="primary", width="stretch")

        if tombol_prediksi:
            st.session_state['prediksi_aktif'] = True

        st.markdown("---")

        # ---- BAGIAN 2: DATA BALITA (kiri) | HASIL PREDIKSI (kanan) ----
        col_data, col_hasil = st.columns(2, gap="large")

        if st.session_state.get('prediksi_aktif', False):
            # Encode & Prediksi
            jk_encoded = mapping_jk[jenis_kelamin]
            input_data = pd.DataFrame([{
                "Usia (Bulan)"         : usia,
                "Jenis_Kelamin_Encoded": jk_encoded,
                "Tinggi Badan (cm)"    : tinggi_badan,
                "Berat Badan (kg)"     : berat_badan
            }])
            prediksi_kode  = int(model.predict(input_data)[0])
            prediksi_label = inverse_label[prediksi_kode]
            probabilitas   = model.predict_proba(input_data)[0]

            with col_data:
                st.markdown("#### 📋 Data Balita")
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("**👶 Usia**")
                    st.markdown(f"""<div style="border: 2px solid #888; padding: 8px 12px; border-radius: 6px; font-size: 15px;">{usia} Bulan</div>""", unsafe_allow_html=True)
                with d2:
                    st.markdown("**⚧ Gender**")
                    st.markdown(f"""<div style="border: 2px solid #888; padding: 8px 12px; border-radius: 6px; font-size: 15px;">{jenis_kelamin}</div>""", unsafe_allow_html=True)
                st.markdown("")
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

            class_idx_map = {int(cls_val): idx for idx, cls_val in enumerate(model.classes_)}

            with col_chart:
                st.markdown("#### 📈 Tingkat Keyakinan Model")
                nama_kelas = ["Normal", "Stunted", "Severely Stunted"]
                for kelas in nama_kelas:
                    cls_kode = mapping_label[kelas]
                    idx = class_idx_map.get(cls_kode, 0)
                    prob = probabilitas[idx]
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
                prob_normal = probabilitas[class_idx_map.get(mapping_label["Normal"], 0)]
                prob_stunted = probabilitas[class_idx_map.get(mapping_label["Stunted"], 0)]
                prob_severe = probabilitas[class_idx_map.get(mapping_label["Severely Stunted"], 0)]

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
                if chart_img is not None:
                    st.image(chart_img, width="stretch")

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
                            width="stretch"
                        )
                else:
                    st.info("💡 Gambar grafik WHO tidak ditemukan di folder assets/images.")
            except Exception as e:
                st.error(f"Gagal memuat grafik pertumbuhan: {str(e)}")

            # ============================================================
            # SECTION: ANALISIS Z-SCORE & SD DI BAWAH GRAFIK
            # ============================================================
            h_data_chart = WHO_HAZ.get(jenis_kelamin, {}).get(usia)
            w_data_chart = WHO_WFA.get(jenis_kelamin, {}).get(usia)

            if h_data_chart and w_data_chart:
                # ---------- TINGGI BADAN (HAZ) ----------
                M_h, SD_h = h_data_chart
                z_score_h = hitung_haz(tinggi_badan, M_h, SD_h)

                # Ambang batas DIHITUNG dari M & SD, bukan diketik manual
                h_med = round(M_h, 1)
                h_m2 = round(M_h - 2 * SD_h, 1)
                h_m3 = round(M_h - 3 * SD_h, 1)

                # ---------- BERAT BADAN (WAZ) ----------
                L_w, M_w, S_w = w_data_chart
                z_score_w = hitung_waz(berat_badan, L_w, M_w, S_w)

                # Ambang batas BB/U bersifat asimetris (L != 1)
                w_med = round(M_w, 1)
                w_m2 = round(nilai_pada_z(-2, L_w, M_w, S_w), 1)
                w_m3 = round(nilai_pada_z(-3, L_w, M_w, S_w), 1)

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

                st.markdown("#### 📊 Posisi Z-Score pada Skala SD (Tinggi Badan / Usia)")

                gauge_z = max(-4, min(4, z_score_h))
                gauge_pct = ((gauge_z + 4) / 8) * 100

                st.markdown(f"""
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px 24px;">
                    <div style="display: flex; margin-bottom: 6px; font-size: 11px; font-weight: 600;">
                        <span style="color: #ef4444; width: 12.5%; text-align: center;">Severely Stunted</span>
                        <span style="color: #f59e0b; width: 12.5%; text-align: center;">Stunted</span>
                        <span style="color: #22c55e; width: 50%; text-align: center;">Normal</span>
                        <span style="color: #3b82f6; width: 25%; text-align: center;">Tinggi</span>
                    </div>
                    <div style="position: relative; height: 32px; border-radius: 16px; overflow: hidden; 
                                background: linear-gradient(to right, 
                                    #fecaca 0%, #fecaca 12.5%, 
                                    #fef08a 12.5%, #fef08a 25%, 
                                    #bbf7d0 25%, #bbf7d0 75%, 
                                    #bfdbfe 75%, #bfdbfe 100%);">
                        <div style="position: absolute; top: -4px; left: calc({gauge_pct:.1f}% - 12px); 
                                    width: 24px; height: 40px; display: flex; flex-direction: column; align-items: center; z-index: 10;">
                            <div style="width: 4px; height: 40px; background: #1e293b; border-radius: 2px;"></div>
                        </div>
                        <div style="position: absolute; top: 50%; left: {gauge_pct:.1f}%; transform: translate(-50%, -50%); 
                                    width: 20px; height: 20px; background: {zscore_color}; border: 3px solid white; 
                                    border-radius: 50%; box-shadow: 0 2px 6px rgba(0,0,0,0.3); z-index: 11;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 6px; font-size: 12px; color: #64748b; font-weight: 500;">
                        <span>-4 SD</span>
                        <span style="position: relative; left: 2%;">-3 SD</span>
                        <span style="position: relative; left: 1%;">-2 SD</span>
                        <span>0</span>
                        <span style="position: relative; right: 1%;">+2 SD</span>
                        <span style="position: relative; right: 2%;">+3 SD</span>
                        <span>+4 SD</span>
                    </div>
                    <div style="text-align: center; margin-top: 12px; padding: 8px 16px; background: white; border-radius: 8px; border: 1px solid #e2e8f0;">
                        <span style="font-size: 14px;">Posisi anak: <strong style="color: {zscore_color}; font-size: 16px;">{z_score_h:+.2f} SD</strong></span>
                        <span style="margin-left: 12px; font-size: 13px; color: #888;">|</span>
                        <span style="margin-left: 12px; font-size: 14px;">Klasifikasi: <strong style="color: {zscore_color};">{zscore_status}</strong></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("")

                st.markdown("#### 📏 Detail Perbandingan Tinggi Badan dengan Standar WHO")

                selisih_median = tinggi_badan - h_med
                selisih_m2 = tinggi_badan - h_m2
                selisih_m3 = tinggi_badan - h_m3

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

                st.markdown("#### ⚖️ Detail Perbandingan Berat Badan dengan Standar WHO")

                selisih_bb_med = berat_badan - w_med
                selisih_bb_m2 = berat_badan - w_m2
                selisih_bb_m3 = berat_badan - w_m3

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

                if tinggi_badan >= h_med:
                    summary_parts.append(f"📏 **Tinggi Badan:** Mencapai atau melebihi median WHO ({h_med} cm). Pertumbuhan tinggi badan **sangat baik**.")
                elif tinggi_badan >= h_m2:
                    summary_parts.append(f"📏 **Tinggi Badan:** Masih dalam rentang normal WHO, namun **{abs(selisih_median):.1f} cm di bawah** median ({h_med} cm). Pertumbuhan perlu terus dioptimalkan dengan asupan gizi yang baik.")
                elif tinggi_badan >= h_m3:
                    summary_parts.append(f"📏 **Tinggi Badan:** Berada di bawah batas -2 SD ({h_m2} cm), yaitu **{abs(selisih_median):.1f} cm di bawah** median WHO. Anak terindikasi **stunted** dan memerlukan peningkatan asupan gizi serta pemantauan pertumbuhan secara rutin.")
                else:
                    summary_parts.append(f"📏 **Tinggi Badan:** Berada di bawah batas -3 SD ({h_m3} cm), yaitu **{abs(selisih_median):.1f} cm di bawah** median WHO. Anak terindikasi **severely stunted** dan memerlukan penanganan medis segera.")

                summary_parts.append("")

                if berat_badan >= w_med:
                    summary_parts.append(f"⚖️ **Berat Badan:** Mencapai atau melebihi median WHO ({w_med} kg). Berat badan **sangat baik**.")
                elif berat_badan >= w_m2:
                    summary_parts.append(f"⚖️ **Berat Badan:** Masih dalam rentang normal, namun **{abs(selisih_bb_med):.1f} kg di bawah** median WHO ({w_med} kg). Perlu dipantau agar tidak terus menurun, perbanyak asupan protein dan kalori.")
                elif berat_badan >= w_m3:
                    summary_parts.append(f"⚖️ **Berat Badan:** Di bawah batas -2 SD ({w_m2} kg). Anak terindikasi **underweight** (kurus) dan memerlukan peningkatan asupan gizi, terutama protein hewani dan lemak sehat.")
                else:
                    summary_parts.append(f"⚖️ **Berat Badan:** Di bawah batas -3 SD ({w_m3} kg). Anak terindikasi **severely underweight** (sangat kurus) dan memerlukan penanganan medis segera serta program gizi intensif.")

                st.info("\n".join(summary_parts))

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

# ============================================================
# DISCLAIMER MEDIS & FOOTER
# ============================================================
st.markdown("")
st.markdown("""
<div style="background: #fffbeb; border: 1px solid #fcd34d; border-radius: 12px; padding: 20px 24px; margin-top: 32px; margin-bottom: 24px;">
    <p style="margin: 0 0 8px 0; font-size: 16px; font-weight: 700; color: #b45309;">
        ⚠️ Disclaimer Medis
    </p>
    <p style="margin: 0; font-size: 13.5px; color: #78350f; line-height: 1.6;">
        Aplikasi ini merupakan <strong>alat bantu deteksi dini</strong> dan <strong>bukan pengganti diagnosis medis profesional</strong>. Hasil prediksi harus selalu dikonfirmasi oleh tenaga kesehatan yang kompeten (dokter anak / petugas puskesmas). Pengembang tidak bertanggung jawab atas keputusan medis yang diambil hanya berdasarkan output aplikasi ini.
    </p>
</div>
<p style="text-align: center; color: #94a3b8; font-size: 12.5px; margin-top: 16px; margin-bottom: 24px;">
    © 2025 — Sistem Deteksi Dini Stunting | Penelitian Skripsi
</p>
""", unsafe_allow_html=True)

