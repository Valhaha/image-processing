# -*- coding: utf-8 -*-
"""
=====================================================================
 期中專題：基於 AI 輔助之互動式影像處理與人臉分析工具
 --------------------------------------------------------------------
 執行方式（本機）：
      streamlit run app.py
 檔案結構：
      app.py              -> 主程式（含四大核心功能）
      requirements.txt    -> 相依套件清單
      generate_report.py  -> PDF 報告產生腳本
 --------------------------------------------------------------------
 四大核心功能模組：
      功能 1  影像載入與基礎解析（PNG / JPG / BMP + 中繼資訊）
      功能 2  多色彩空間轉換與同步直方圖分析（Original / RGB / YUV）
      功能 3  自訂卷積核空間濾波與 PSNR 量化評估
      功能 4  人臉偵測模組（Haar-Cascade + 信心度）
 --------------------------------------------------------------------
 使用之第三方套件：
      streamlit   網頁互動介面（GUI）
      opencv      影像處理 / 色彩空間 / 卷積 / 人臉偵測
      numpy       數值運算（矩陣、直方圖、PSNR）
      pillow      影像格式 I/O
      matplotlib  直方圖繪圖
=====================================================================
"""

import math

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import matplotlib

# 使用無互動式後端，避免在 Streamlit 內彈出獨立視窗
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# 全域頁面設定
# =====================================================================
st.set_page_config(
    page_title="互動式影像處理與人臉分析工具",
    page_icon="🖼️",
    layout="wide",
)


# =====================================================================
# 工具函式區
# =====================================================================


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """
    將 OpenCV 的 BGR 色彩空間轉為 RGB（供 Streamlit / PIL 顯示）。
    因為 OpenCV 讀圖預設為 BGR，而多數顯示函式（st.image、PIL）使用 RGB，
    因此顯示前必須轉換，否則顏色會「藍紅互換」。
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    """將 RGB 轉回 BGR（供 OpenCV 處理）。"""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


# =====================================================================
# 功能 1：影像載入與基礎解析
# =====================================================================


def load_image(uploaded_file) -> tuple:
    """
    讀取上傳的點陣圖檔（PNG / JPG / JPEG / BMP），
    並解析影像的中繼資訊（解析度、色彩維度、檔案大小）。
    回傳 (BGR 影像 ndarray, 中繼資訊 dict)。
    """
    # 讀取檔案位元內容，再透過 OpenCV 解碼，可支援各種點陣圖格式
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # 一律以彩色讀入

    if img_bgr is None:
        st.error("❌ 無法讀取此影像檔案，請確認格式為 PNG / JPG / JPEG / BMP。")
        st.stop()

    # ---- 解析中繼資訊 -------------------------------------------------
    height, width, channels = img_bgr.shape  # 高、寬、色彩維度
    file_size = round(uploaded_file.size / (1024 * 1024), 3)  # 檔案大小（MB）

    meta = {
        "檔名": uploaded_file.name,
        "解析度 (寬 x 高)": f"{width} x {height}",
        "色彩維度 (通道數)": channels,
        "色彩空間": "BGR（OpenCV 預設）",
        "檔案大小 (MB)": file_size,
        "資料型態": str(img_bgr.dtype),
    }
    return img_bgr, meta


def generate_synthetic_image(width: int = 640, height: int = 480) -> np.ndarray:
    """
    產生一張「合成測試影像」，內含：
      - 水平灰階漸層背景
      - 彩色圓形與矩形（方便觀察 R/G/B 通道與直方圖分布）
    此函式目的：讓使用者在未上傳影像時，也能離線測試功能 1~3。
    """
    # 建立灰階漸層底圖（0~255 水平漸層），再複製到三個通道
    grad = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    img = cv2.merge([grad, grad, grad])  # 轉為 BGR 三通道

    # 繪製多個彩色幾何圖形，產生不同強度分布
    cv2.circle(img, (180, 180), 100, (0, 0, 255), -1)    # 紅色圓（BGR）
    cv2.circle(img, (460, 140), 90, (0, 255, 0), -1)     # 綠色圓
    cv2.rectangle(img, (120, 300), (320, 440), (255, 0, 0), -1)  # 藍色矩形
    cv2.rectangle(img, (380, 320), (560, 440), (0, 255, 255), -1)  # 黃色矩形
    cv2.putText(img, "Synthetic Test Image", (150, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


# =====================================================================
# 功能 2：多色彩空間轉換與同步直方圖分析
# =====================================================================


def convert_color_space(img_bgr: np.ndarray, mode: str) -> tuple:
    """
    依使用者選擇的模式回傳「要顯示的影像」與「各通道強度陣列」。

    模式說明：
      Original : 原始 BGR -> 顯示 RGB
      R / G / B: 分離各通道，單一通道灰階圖
      YUV      : 轉換至 YUV 色彩空間，Y=亮度、U/V=色度
    回傳 (display_img, channels_dict)，其中 channels_dict 的 key 為通道名稱、
    value 為單通道的灰階 ndarray，供直方圖繪製使用。
    """
    if mode == "Original":
        display = bgr_to_rgb(img_bgr)
        channels = {"R": img_bgr[:, :, 2], "G": img_bgr[:, :, 1], "B": img_bgr[:, :, 0]}

    elif mode == "R 分離頻道":
        r = img_bgr[:, :, 2]
        display = cv2.merge([r, r, r])          # 複製到三通道顯示灰階
        display = bgr_to_rgb(display)
        channels = {"R": r}

    elif mode == "G 分離頻道":
        g = img_bgr[:, :, 1]
        display = cv2.merge([g, g, g])
        display = bgr_to_rgb(display)
        channels = {"G": g}

    elif mode == "B 分離頻道":
        b = img_bgr[:, :, 0]
        display = cv2.merge([b, b, b])
        display = bgr_to_rgb(display)
        channels = {"B": b}

    elif mode == "YUV":
        # BGR -> YUV。OpenCV 使用 YCrCb 型 YUV 定義（Y 亮度、U/Cb 色度、V/Cr 色度）
        yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        y = yuv[:, :, 0]
        u = yuv[:, :, 1]
        v = yuv[:, :, 2]
        # Y 顯示灰階；U、V 各別以灰階呈現其強度分布
        display = cv2.merge([y, y, y])
        display = bgr_to_rgb(display)
        channels = {"Y": y, "U": u, "V": v}

    else:  # 預設回到原始
        display = bgr_to_rgb(img_bgr)
        channels = {"R": img_bgr[:, :, 2], "G": img_bgr[:, :, 1], "B": img_bgr[:, :, 0]}

    return display, channels


def compute_histogram(channel: np.ndarray) -> np.ndarray:
    """
    計算單通道灰階影像的直方圖。
    橫軸為像素強度 0~255，縱軸為該強度的像素數量。
    使用 OpenCV 的 calcHist 計算，再壓平為長度 256 的陣列。
    """
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    return hist.flatten()


def plot_histogram(channels: dict) -> plt.Figure:
    """
    使用 matplotlib 繪製各通道的直方圖。
    每個通道以不同顏色線條疊加，橫軸 0~255、縱軸像素數量。
    回傳 Figure 物件供 st.pyplot 顯示。
    """
    # 通道對應顯示顏色（Y/U/V 使用灰/綠/藍做視覺區別）
    color_map = {
        "R": "red", "G": "green", "B": "blue",
        "Y": "gray", "U": "lime", "V": "cyan",
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, ch in channels.items():
        hist = compute_histogram(ch)
        ax.plot(hist, color=color_map.get(name, "black"), label=name, linewidth=1.5)
    ax.set_xlim([0, 255])                # 橫軸固定 0~255
    ax.set_xlabel("Pixel Intensity (0-255)")
    ax.set_ylabel("Pixel Count")
    ax.set_title("Histogram")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# =====================================================================
# 功能 3：自訂卷積核空間濾波與 PSNR 量化評估
# =====================================================================


def parse_kernel(text: str) -> np.ndarray:
    """
    解析使用者輸入的二維卷積核字串。
    支援以「;」分隔列、以「,」或空白分隔欄位，例如：
        3x3 銳化核："0,-1,0; -1,5,-1; 0,-1,0"
        5x5 高斯核："1,4,6,4,1; 4,16,24,16,4; ..."
    回傳 float32 的二維 numpy 陣列。
    """
    rows = [r for r in text.strip().split(";") if r.strip() != ""]
    matrix = []
    for r in rows:
        # 同時支援逗號或空白分隔
        vals = r.replace(",", " ").split()
        if not vals:
            continue
        matrix.append([float(v) for v in vals])
    kernel = np.array(matrix, dtype=np.float32)
    return kernel


def validate_kernel(kernel: np.ndarray) -> str:
    """
    驗證卷積核是否合法，回傳錯誤訊息（空字串表示合法）。
    需檢查：
      1. 是否為空
      2. 是否為二維
      3. 是否為方陣（3x3 / 5x5 等）
      4. 是否奇數邊長（卷積需有明確中心點）
      5. 是否包含非數值（NaN/Inf）
    """
    if kernel.size == 0:
        return "卷積核不可為空。"
    if kernel.ndim != 2:
        return "卷積核必須為二維矩陣。"
    if kernel.shape[0] != kernel.shape[1]:
        return f"卷積核必須為方陣，目前為 {kernel.shape[0]}x{kernel.shape[1]}。"
    if kernel.shape[0] % 2 == 0:
        return "卷積核邊長必須為奇數（如 3x3、5x5），以具備中心點。"
    if not np.isfinite(kernel).all():
        return "卷積核包含非數值（NaN 或 Inf），請檢查輸入。"
    return ""


def apply_convolution(img_bgr: np.ndarray, kernel: np.ndarray, norm_mode: str) -> np.ndarray:
    """
    對輸入影像執行卷積濾波。
    norm_mode 決定正規化方式：
      - 無正規化   ：直接使用原核
      - 除以核總和 ：kernel / sum(kernel)，用於模糊/平滑（需防除以零）
      - 除以絕對值總和：kernel / sum(|kernel|)，避免負數核互相抵消
    使用 OpenCV filter2D 加速（與手寫雙層迴圈卷積等價，效能更高）。
    """
    k = kernel.copy()

    # ---- 正規化處理 ----------------------------------------------------
    if norm_mode == "除以核總和":
        s = float(np.sum(k))
        if abs(s) < 1e-12:  # 防除以零：核總和為 0（如邊緣偵測核）
            st.warning("⚠️ 核總和為 0，無法除以核總和，已自動改用「除以絕對值總和」。")
            k = k / (float(np.sum(np.abs(k))) + 1e-12)
        else:
            k = k / s
    elif norm_mode == "除以絕對值總和":
        k = k / (float(np.sum(np.abs(k))) + 1e-12)
    # 無正規化：維持原核

    # ---- 執行卷積 ------------------------------------------------------
    # ddepth=-1 表示輸出與輸入相同資料型態；borderType 為邊界填補方式
    filtered = cv2.filter2D(img_bgr, ddepth=-1, kernel=k, borderType=cv2.BORDER_DEFAULT)
    return filtered


def compute_psnr(original: np.ndarray, filtered: np.ndarray) -> float:
    """
    計算峰值信噪比（PSNR，單位 dB）。
    公式：
        MSE = (1/(m*n*c)) * ΣΣΣ ( I - K )^2
        PSNR = 10 * log10( MAX_I^2 / MSE )
    其中 MAX_I = 255（8-bit 影像最大強度）。
    若兩圖完全相同，MSE = 0，PSNR 趨近無限大（回傳 float('inf')）。
    """
    original = original.astype(np.float64)
    filtered = filtered.astype(np.float64)
    mse = float(np.mean((original - filtered) ** 2))  # 平均平方誤差
    if mse < 1e-12:
        return float("inf")  # 完全一致 -> PSNR 無限大
    max_i = 255.0
    psnr = 10.0 * math.log10((max_i ** 2) / mse)
    return psnr


# =====================================================================
# 功能 4：人臉偵測模組（Haar-Cascade + 信心度）
# =====================================================================


@st.cache_resource
def load_face_cascade():
    """
    載入 OpenCV 內建的 Haar-Cascade 人臉偵測器（正面人臉）。
    使用 st.cache_resource 快取，避免每次 rerun 都重新載入。
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def detect_faces(img_bgr: np.ndarray):
    """
    使用 Haar-Cascade 偵測人臉。
    呼叫 detectMultiScale3 取得：
        faces        -> 每張人臉的 (x, y, w, h) 邊界框
        rejectLevels -> 偵測器的拒絕層級
        levelWeights -> 各物件的權重（可換算成信心度）
    回傳 (annotated_img, faces, confidences)。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # 直方圖等化，提升對比、強化偵測率

    cascade = load_face_cascade()
    # scaleFactor：每次掃描縮小比例；minNeighbors：降低誤判的鄰居門檻
    faces, reject_levels, level_weights = cascade.detectMultiScale3(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        outputRejectLevels=True,
    )

    # ---- 將 levelWeights 換算為 0~1 的信心度 -------------------------
    # levelWeights 為 Haar 偵測器的決策權重，數值越大代表越像人臉。
    # 使用 sigmoid 函式映射至 (0,1) 作為信心度估計。
    confidences = []
    for w in level_weights:
        conf = 1.0 / (1.0 + math.exp(-float(w)))  # sigmoid
        confidences.append(round(conf, 3))

    # ---- 在影像上繪製邊界框與信心度 -----------------------------------
    annotated = img_bgr.copy()
    for (x, y, w, h), conf in zip(faces, confidences):
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{conf:.2f}"
        cv2.putText(annotated, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return annotated, faces, confidences


# =====================================================================
# 主程式進入點（Streamlit 版面）
# =====================================================================


def main():
    """Streamlit 主程式：組裝四大功能與側欄輸入。"""
    st.title("🖼️ 互動式影像處理與人臉分析工具")
    st.caption("基於生成式 AI 輔助開發 | Streamlit + OpenCV | 期中專題")

    # ------------------------------------------------------------------
    # 側欄：影像載入
    # ------------------------------------------------------------------
    st.sidebar.header("📂 影像載入")
    uploaded = st.sidebar.file_uploader(
        "上傳影像（PNG / JPG / JPEG / BMP）",
        type=["png", "jpg", "jpeg", "bmp"],
    )

    use_synthetic = st.sidebar.button("使用合成測試影像")

    img_bgr = None
    meta = None

    if uploaded is not None:
        img_bgr, meta = load_image(uploaded)
    elif use_synthetic:
        img_bgr = generate_synthetic_image()
        meta = {
            "檔名": "synthetic_test.png",
            "解析度 (寬 x 高)": f"{img_bgr.shape[1]} x {img_bgr.shape[0]}",
            "色彩維度 (通道數)": img_bgr.shape[2],
            "色彩空間": "BGR（合成影像）",
            "檔案大小 (MB)": "N/A（記憶體產生）",
            "資料型態": str(img_bgr.dtype),
        }
    else:
        st.info("👈 請於左側上傳影像，或點擊「使用合成測試影像」開始。")
        st.stop()

    # ------------------------------------------------------------------
    # 分頁：四大核心功能
    # ------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["1️⃣ 影像載入與解析", "2️⃣ 色彩空間與直方圖", "3️⃣ 卷積濾波與 PSNR", "4️⃣ 人臉偵測"]
    )

    # ============ 功能 1：影像載入與基礎解析 ============================
    with tab1:
        st.subheader("功能 1｜影像載入與基礎解析")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(bgr_to_rgb(img_bgr), caption="原始影像", width="stretch")
        with c2:
            st.markdown("#### 影像中繼資訊")
            # 以 markdown 表格呈現中繼資訊（避免額外引入 pandas 相依套件）
            table_md = "| 項目 | 數值 |\n|---|---|\n"
            for key, val in meta.items():
                table_md += f"| {key} | {val} |\n"
            st.markdown(table_md)

    # ============ 功能 2：色彩空間與直方圖 ==============================
    with tab2:
        st.subheader("功能 2｜多色彩空間轉換與同步直方圖分析")
        mode = st.radio(
            "選擇檢視模式",
            ["Original", "R 分離頻道", "G 分離頻道", "B 分離頻道", "YUV"],
            horizontal=True,
        )
        display_img, channels = convert_color_space(img_bgr, mode)
        cc1, cc2 = st.columns([1, 1])
        with cc1:
            st.image(display_img, caption=f"顯示模式：{mode}", width="stretch")
        with cc2:
            fig = plot_histogram(channels)
            st.pyplot(fig)
            plt.close(fig)

    # ============ 功能 3：卷積濾波與 PSNR ===============================
    with tab3:
        st.subheader("功能 3｜自訂卷積核空間濾波與 PSNR 量化評估")

        # 提供常用範例核按鈕，方便快速套用
        st.markdown("##### 快速套用常用卷積核")
        ex1, ex2, ex3, ex4 = st.columns(4)
        preset = None
        if ex1.button("邊緣銳化 (3x3)"):
            preset = "0,-1,0; -1,5,-1; 0,-1,0"
        if ex2.button("高斯平滑 (3x3)"):
            preset = "1,2,1; 2,4,2; 1,2,1"
        if ex3.button("浮雕 (3x3)"):
            preset = "-2,-1,0; -1,1,1; 0,1,2"
        if ex4.button("均值模糊 (3x3)"):
            preset = "1,1,1; 1,1,1; 1,1,1"

        kernel_text = st.text_area(
            "輸入卷積核矩陣（以「;」分隔列，以「,」分隔欄位）",
            value=preset if preset else "0,-1,0; -1,5,-1; 0,-1,0",
            height=120,
        )
        norm_mode = st.selectbox(
            "正規化（Normalization）方式",
            ["無正規化", "除以核總和", "除以絕對值總和"],
        )

        # 解析與驗證卷積核
        kernel = parse_kernel(kernel_text)
        err = validate_kernel(kernel)
        if err:
            st.error(f"❌ 卷積核錯誤：{err}")
        else:
            filtered = apply_convolution(img_bgr, kernel, norm_mode)
            psnr = compute_psnr(img_bgr, filtered)

            st.markdown(f"**目前卷積核大小：** {kernel.shape[0]}×{kernel.shape[1]}")
            fc1, fc2 = st.columns([1, 1])
            with fc1:
                st.image(bgr_to_rgb(img_bgr), caption="濾波前（原始）", width="stretch")
            with fc2:
                st.image(bgr_to_rgb(filtered), caption="濾波後", width="stretch")

            # 顯示 PSNR 數值
            st.metric("PSNR (dB)", "無限大" if math.isinf(psnr) else round(psnr, 2))

    # ============ 功能 4：人臉偵測 ======================================
    with tab4:
        st.subheader("功能 4｜人臉偵測模組（Haar-Cascade）")
        if st.button("執行人臉偵測"):
            annotated, faces, confidences = detect_faces(img_bgr)
            d1, d2 = st.columns([1, 1])
            with d1:
                st.image(bgr_to_rgb(annotated), caption="偵測結果（含邊界框）",
                         width="stretch")
            with d2:
                st.markdown("#### 偵測結果")
                st.metric("偵測到的人臉數量", len(faces))
                if len(faces) > 0:
                    for i, ((x, y, w, h), c) in enumerate(zip(faces, confidences), start=1):
                        st.write(f"人臉 {i}：位置=({x},{y}) 大小={w}x{h} 信心度={c}")
                else:
                    st.info("未偵測到人臉。請上傳清晰且光線充足的正面人臉照片。")
        else:
            st.info("點擊上方按鈕開始偵測。多人臉場景請上傳含多張人臉的合照。")


# =====================================================================
# 程式進入點
# =====================================================================
if __name__ == "__main__":
    main()
