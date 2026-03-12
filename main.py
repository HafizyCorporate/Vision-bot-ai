import os
import base64
import cv2
import numpy as np
import telebot
import threading
from flask import Flask, request, jsonify
from ultralytics import YOLO

# --- 1. SETUP VARIABEL & BOT ---
# Ambil rahasia dari environment Railway
TOKEN = os.environ.get("BOT_TOKEN", "TOKEN_BOT_KAMU")
CHAT_ID = os.environ.get("CHAT_ID", "CHAT_ID_KAMU_BERUPA_ANGKA")
PORT = int(os.environ.get("PORT", 8080))

bot = telebot.TeleBot(TOKEN)
app = Flask(__name__)

# --- 2. LOAD OTAK AI ---
print("Memuat model YOLOv8 Nano...")
model = YOLO('yolov8n.pt') # Pakai versi Nano agar server Railway gratisan tidak RAM bocor
# --- 3. GERBANG PENERIMA FOTO DARI NODE.JS ---
@app.route('/vision', methods=['POST'])
def vision_endpoint():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "Data gambar kosong"}), 400
        
    try:
        image_b64 = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_b64)
        image_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
        
        print("[VISION] Menganalisis frame...")
        results = model(img)
        
        deteksi_ada = False
        for r in results:
            if len(r.boxes) > 0:
                deteksi_ada = True
                break
                
        # Gambar kotak merah/label di foto hasil (kalau ada)
        res_plotted = results[0].plot()
        _, buffer = cv2.imencode('.jpg', res_plotted)
        foto_final = buffer.tobytes()

        # SEKARANG KITA PAKSA DIA SELALU KIRIM KE TELEGRAM!
        if deteksi_ada:
            bot.send_photo(CHAT_ID, foto_final, caption="⚠️ [ALERT] ADA OBJEK TERDETEKSI!")
            print("[VISION] Ancaman dilaporkan ke Markas!")
        else:
            bot.send_photo(CHAT_ID, foto_final, caption="✅ [INFO] Area terpantau aman (Tidak ada objek).")
            print("[VISION] Foto aman dilaporkan ke Markas!")
            
        return jsonify({"status": "Laporan terkirim"}), 200
            
    except Exception as e:
        print(f"Error sistem vision: {e}")
        return jsonify({"error": "Otak AI gagal memproses"}), 500

# --- 4. FUNGSI REMOTE CONTROL (TELEGRAM) ---
@bot.message_handler(commands=['land'])
def command_land(message):
    bot.reply_to(message, "🔴 PERINTAH DITERIMA: Wahana mendarat darurat sekarang!")
    # (Di pengembangan selanjutnya, bot ini bisa nembak API balik ke Node.js)

# --- 5. MESIN PENGGERAK GANDA (THREADING) ---
def run_bot():
    bot.infinity_polling()

if __name__ == '__main__':
    # Jalankan Bot Telegram di jalur belakang (Background Thread)
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.start()
    
    # Jalankan API Penerima Foto di jalur utama
    print(f"🔥 Server Vision API aktif di port {PORT}...")
    app.run(host="0.0.0.0", port=PORT)
