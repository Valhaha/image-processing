# 互動式影像處理與人臉分析工具

基於生成式 AI 輔助之互動式影像處理與人臉分析工具（期中專題）。

以 **Streamlit** 打造互動網頁介面，整合 **OpenCV** 與 **NumPy** 實作四大核心功能。

## 功能

1. **影像載入與基礎解析** — 支援 PNG / JPG / JPEG / BMP，顯示解析度、色彩維度、檔案大小。
2. **多色彩空間轉換與同步直方圖** — Original / R / G / B 分離頻道 / YUV 檢視，並同步繪製對應通道直方圖（0–255）。
3. **自訂卷積核空間濾波與 PSNR** — 動態輸入任意方陣卷積核（銳化、高斯平滑、浮雕等）、正規化選項，即時計算 PSNR（dB）。
4. **人臉偵測模組** — Haar-Cascade 偵測人臉，標記邊界框並回報數量與信心度。

## 安裝

```bash
pip install -r requirements.txt
```

## 執行

```bash
streamlit run app.py
```

## 專案結構

```
├── app.py            # 主程式（四大核心功能，含詳細中文註解）
├── requirements.txt  # 相依套件清單
├── README.md         # 專案說明
└── .gitignore
```

## 部署（Streamlit Cloud）

1. 將本儲存庫上傳至 GitHub（含 app.py 與 requirements.txt）。
2. 至 [streamlit.io/cloud](https://streamlit.io/cloud) 以 GitHub 登入，點選 **New app**。
3. 選取本儲存庫與 `app.py`，首次部署會自動安裝相依套件。
4. 於 App 設定將權限設為 **Public**（公開）。

## 技術棧

- Python 3.9+
- Streamlit、OpenCV、NumPy、Pillow、Matplotlib
