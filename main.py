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
model = YOLO('helmet.pt') 

# --- 3. FITUR BARU: RENDER VIDEO VIA TELEGRAM ---
@bot.message_handler(content_types=['video', 'document'])
def handle_video(message):
    bot.reply_to(message, "⚙️ [SYSTEM] Video diterima! Memulai proses rendering AI. Harap tunggu, ini membutuhkan komputasi berat (1-3 menit)...")
    
    try:
        # 1. Download Video dari Telegram
        if message.content_type == 'video':
            file_info = bot.get_file(message.video.file_id)
        else:
            file_info = bot.get_file(message.document.file_id)
            
        downloaded_file = bot.download_file(file_info.file_path)
        
        input_path = "temp_input.mp4"
        output_path = "temp_output.mp4"
        
        with open(input_path, 'wb') as new_file:
            new_file.write(downloaded_file)
            
        # 2. Buka Video pakai OpenCV
        cap = cv2.VideoCapture(input_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 3. Siapkan Mesin Penjahit Video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # 4. Bedah & Render Frame per Frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Masukkan ke Otak AI (conf=0.15 biar sensitif nangkap yang di belakang)
            results = model(frame, conf=0.15, verbose=False)
            res_plotted = results[0].plot()
            
            # Jahit kembali jadi video
            out.write(res_plotted)
            
        cap.release()
        out.release()
        
        # 5. Kirim Balik Hasilnya ke Telegram
        with open(output_path, 'rb') as video_file:
            bot.send_video(message.chat.id, video_file, caption="✅ [ETLE SUCCESS] Rendering Video Selesai!\nSemua target telah ditandai.")
            
        # Bersihkan file sampah
        os.remove(input_path)
        os.remove(output_path)
        
    except Exception as e:
        bot.reply_to(message, f"❌ [ERROR] Gagal memproses video: {e}")
        print(f"Error Video: {e}")

# --- 4. GERBANG PENERIMA VISION (TETAP ADA BUAT WEB HTML) ---
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
        
        # Eksekusi AI untuk Web (Sensitivitas tinggi)
        results = model(img, conf=0.15)
        
        pelanggaran_helm = False
        daftar_objek = []
        
        for r in results:
            for box in r.boxes:
                kelas_id = int(box.cls[0])
                nama_objek = model.names[kelas_id].lower() 
                daftar_objek.append(nama_objek)
                if "no" in nama_objek or "without" in nama_objek or "bare" in nama_objek:
                    pelanggaran_helm = True
                
        res_plotted = results[0].plot()
        _, buffer = cv2.imencode('.jpg', res_plotted)
        foto_final = buffer.tobytes()

        if pelanggaran_helm:
            pesan = "🚨 [ETLE ALERT] PELANGGARAN TERDETEKSI!\n❌ Pengendara terpantau TIDAK MENGGUNAKAN HELM."
            bot.send_photo(CHAT_ID, foto_final, caption=pesan)
            
        return jsonify({"status": "Diproses"}), 200
            
    except Exception as e:
        return jsonify({"error": "Server error"}), 500

# --- 5. FUNGSI TELEGRAM & RUNNER ---
@bot.message_handler(commands=['start', 'land'])
def command_land(message):
    bot.reply_to(message, "🔴 PERINTAH DITERIMA: Sistem ETLE Siaga!\n\nKirimkan file VIDEO ke sini, dan AI akan merendernya.")

def run_bot():
    bot.infinity_polling()

if __name__ == '__main__':
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.start()
    print(f"🔥 Server ETLE aktif di port {PORT}...")
    app.run(host="0.0.0.0", port=PORT)
