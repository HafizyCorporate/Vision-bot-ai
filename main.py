import os
import base64
import cv2
import numpy as np
import telebot
import threading
import easyocr # <-- INI DIA OTAK PEMBACA HURUFNYA
from flask import Flask, request, jsonify
from ultralytics import YOLO

# --- 1. SETUP VARIABEL ---
TOKEN = os.environ.get("BOT_TOKEN", "TOKEN_BOT_KAMU")
CHAT_ID = os.environ.get("CHAT_ID", "CHAT_ID_KAMU_BERUPA_ANGKA")
PORT = int(os.environ.get("PORT", 8080))

bot = telebot.TeleBot(TOKEN)
app = Flask(__name__)

# --- 2. LOAD 2 OTAK AI SEKALIGUS ---
print("Memuat model YOLOv8 Nano...")
model = YOLO('yolov8n.pt')

print("Memuat model OCR Pembaca Plat...")
# Peringatan: Ini lumayan berat buat server gratisan!
reader = easyocr.Reader(['en'], gpu=False) 

# --- 3. GERBANG PENERIMA VISION ---
@app.route('/vision', methods=['POST'])
def vision_endpoint():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "Data kosong"}), 400
        
    try:
        image_b64 = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_b64)
        image_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
        
        results = model(img)
        
        deteksi_kendaraan = False
        plat_terbaca = ""
        
        # Bedah hasil pelihatan YOLO
        for r in results:
            for box in r.boxes:
                kelas_id = int(box.cls[0])
                # ID COCO: 2=Mobil, 3=Motor, 5=Bus, 7=Truk
                if kelas_id in [2, 3, 5, 7]: 
                    deteksi_kendaraan = True
                    
                    # Potong gambar persis di area kendaraan biar OCR fokus
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    kendaraan_img = img[y1:y2, x1:x2]
                    
                    # Suruh OCR membaca teks di potongan gambar itu
                    teks_hasil = reader.readtext(kendaraan_img, detail=0)
                    if teks_hasil:
                        plat_terbaca += " ".join(teks_hasil) + " | "
                
        # Gambar kotak di foto asli
        res_plotted = results[0].plot()
        _, buffer = cv2.imencode('.jpg', res_plotted)
        foto_final = buffer.tobytes()

        # Eksekusi Laporan ke Telegram
        if deteksi_kendaraan:
            pesan = "🚓 [ETLE ALERT] KENDARAAN TERDETEKSI!\n\n"
            if plat_terbaca:
                pesan += f"🔍 Teks Terbaca: {plat_terbaca}"
            else:
                pesan += "❌ Plat nomor tidak terlihat/terbaca."
                
            bot.send_photo(CHAT_ID, foto_final, caption=pesan)
            print("[VISION] Laporan ETLE dikirim!")
        else:
            print("[VISION] Area aman, tidak ada kendaraan.")
            
        return jsonify({"status": "Diproses"}), 200
            
    except Exception as e:
        print(f"Error ETLE: {e}")
        return jsonify({"error": "Server overload"}), 500

# --- 4. FUNGSI TELEGRAM & RUNNER ---
@bot.message_handler(commands=['land'])
def command_land(message):
    bot.reply_to(message, "🔴 PERINTAH DITERIMA: Sistem siaga!")

def run_bot():
    bot.infinity_polling()

if __name__ == '__main__':
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.start()
    app.run(host="0.0.0.0", port=PORT)
