import os
import base64
import cv2
import numpy as np
import telebot
import threading
from flask import Flask, request, jsonify
from ultralytics import YOLO

# --- 1. SETUP VARIABEL ---
TOKEN = os.environ.get("BOT_TOKEN", "TOKEN_BOT_KAMU")
CHAT_ID = os.environ.get("CHAT_ID", "CHAT_ID_KAMU_BERUPA_ANGKA")
PORT = int(os.environ.get("PORT", 8080))

bot = telebot.TeleBot(TOKEN)
app = Flask(__name__)

# --- 2. GANTI OTAK AI KE MODEL HELM ---
print("Memuat model ETLE Khusus Helm...")
# Memakai file helmet.pt yang barusan kamu upload
model = YOLO('helmet.pt') 

# --- 3. GERBANG PENERIMA VISION ---
@app.route('/vision', methods=['POST'])
def vision_endpoint():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "Data kosong"}), 400
        
    try:
        # Bersihkan dan baca foto dari web hijau
        image_b64 = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_b64)
        image_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
        
        # Eksekusi Otak AI
        results = model(img)
        
        ada_deteksi = False
        pelanggaran_helm = False
        daftar_objek = []
        
        # Bedah hasil penglihatan AI Helm
        for r in results:
            if len(r.boxes) > 0:
                ada_deteksi = True
                
            for box in r.boxes:
                kelas_id = int(box.cls[0])
                nama_objek = model.names[kelas_id].lower() 
                daftar_objek.append(nama_objek)
                
                print(f"[VISION] AI melihat: {nama_objek}") 
                
                # Cek apakah terdeteksi kepala tanpa helm (no helmet / without helmet / bare)
                if "no" in nama_objek or "without" in nama_objek or "bare" in nama_objek:
                    pelanggaran_helm = True
                
        # Gambar kotak di foto hasil
        res_plotted = results[0].plot()
        _, buffer = cv2.imencode('.jpg', res_plotted)
        foto_final = buffer.tobytes()

        # Eksekusi Laporan ke Telegram
        if pelanggaran_helm:
            # Skenario 1: Kena Tilang (Nggak pakai helm)
            pesan = "🚨 [ETLE ALERT] PELANGGARAN TERDETEKSI!\n\n"
            pesan += "❌ Pengendara terpantau TIDAK MENGGUNAKAN HELM.\n"
            pesan += f"🔍 Terdeteksi: {', '.join(set(daftar_objek))}"
            bot.send_photo(CHAT_ID, foto_final, caption=pesan)
            print("[VISION] Surat Tilang Helm dikirim ke Telegram!")
            
        elif ada_deteksi:
            # Skenario 2: Aman (Pakai helm, kirim laporan warna hijau)
            pesan = "✅ [ETLE INFO] PENGENDARA TERTIB.\n\n"
            pesan += "👍 Pengendara terpantau MENGGUNAKAN HELM standar.\n"
            pesan += f"🔍 Terdeteksi: {', '.join(set(daftar_objek))}"
            bot.send_photo(CHAT_ID, foto_final, caption=pesan)
            print("[VISION] Laporan Tertib dikirim ke Telegram!")
            
        else:
            # Skenario 3: Foto kosong / jalanan sepi
            print("[VISION] Area aman. Tidak ada pengendara.")
            
        return jsonify({"status": "Diproses"}), 200
            
    except Exception as e:
        print(f"Error Vision: {e}")
        return jsonify({"error": "Server error"}), 500

# --- 4. FUNGSI TELEGRAM & RUNNER ---
@bot.message_handler(commands=['land'])
def command_land(message):
    bot.reply_to(message, "🔴 PERINTAH DITERIMA: Sistem ETLE Siaga!")

def run_bot():
    bot.infinity_polling()

if __name__ == '__main__':
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.start()
    print(f"🔥 Server ETLE aktif di port {PORT}...")
    app.run(host="0.0.0.0", port=PORT)
