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

# --- FUNGSI RAHASIA: TUKANG GAMBAR KOTAK CUSTOM ---
def gambar_custom_kotak(frame, hasil_ai):
    frame_gambar = frame.copy()
    
    # Ambil kotak-kotak hasil deteksi
    for box in hasil_ai[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        kelas_id = int(box.cls[0])
        nama_objek = model.names[kelas_id].lower()
        
        # --- LOGIKA WARNA SYSADMIN ---
        # OpenCV pakai format (B, G, R)
        if "no" in nama_objek or "without" in nama_objek or "bare" in nama_objek:
            warna = (0, 0, 255) # MERAH (Pelanggar)
            label = f"TIDAK PAKAI HELM {conf:.2f}"
        else:
            warna = (0, 255, 0) # HIJAU (Aman/Pakai Helm)
            label = f"PAKAI HELM {conf:.2f}"
            
        # Gambar kotak di sekeliling kepala
        cv2.rectangle(frame_gambar, (x1, y1), (x2, y2), warna, 2)
        
        # Gambar background label biar teksnya jelas
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_gambar, (x1, y1 - 20), (x1 + tw, y1), warna, -1)
        
        # Tulis teks labelnya
        cv2.putText(frame_gambar, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return frame_gambar

# --- 3. FITUR RENDER VIDEO VIA TELEGRAM ---
@bot.message_handler(content_types=['video', 'document'])
def handle_video(message):
    bot.reply_to(message, "⚙️ [SYSTEM] Video diterima! Memulai proses rendering AI...\nSistem akan mewarnai: MERAH (Pelanggar) & HIJAU (Aman).")
    
    try:
        if message.content_type == 'video':
            file_info = bot.get_file(message.video.file_id)
        else:
            file_info = bot.get_file(message.document.file_id)
            
        downloaded_file = bot.download_file(file_info.file_path)
        input_path = "temp_input.mp4"
        output_path = "temp_output.mp4"
        
        with open(input_path, 'wb') as new_file:
            new_file.write(downloaded_file)
            
        cap = cv2.VideoCapture(input_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Tuning AI: conf=0.35 biar nggak asal tilang pejalan kaki
            results = model(frame, conf=0.35, verbose=False)
            
            # PANGGIL TUKANG GAMBAR CUSTOM KITA!
            frame_plotted = gambar_custom_kotak(frame, results)
            
            out.write(frame_plotted)
            
        cap.release()
        out.release()
        
        with open(output_path, 'rb') as video_file:
            bot.send_video(message.chat.id, video_file, caption="✅ [ETLE SUCCESS] Rendering Video Selesai!\n🔴 MERAH = Pelanggar\n🟢 HIJAU = Aman")
            
        os.remove(input_path)
        os.remove(output_path)
        
    except Exception as e:
        bot.reply_to(message, f"❌ [ERROR] Gagal memproses video: {e}")
        print(f"Error Video: {e}")

# --- 4. GERBANG PENERIMA VISION BUAT WEB HTML ---
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
        
        results = model(img, conf=0.35)
        
        pelanggaran_helm = False
        
        for r in results:
            for box in r.boxes:
                kelas_id = int(box.cls[0])
                nama_objek = model.names[kelas_id].lower() 
                if "no" in nama_objek or "without" in nama_objek or "bare" in nama_objek:
                    pelanggaran_helm = True
                
        # Gambar foto untuk web pakai warna custom juga!
        frame_plotted = gambar_custom_kotak(img, results)
        _, buffer = cv2.imencode('.jpg', frame_plotted)
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
    bot.reply_to(message, "🔴 PERINTAH DITERIMA: Sistem ETLE Siaga!\n\nKirimkan file VIDEO, AI akan mewarnai:\n🟢 HIJAU (Aman)\n🔴 MERAH (Melanggar)")

def run_bot():
    bot.infinity_polling()

if __name__ == '__main__':
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.start()
    print(f"🔥 Server ETLE aktif di port {PORT}...")
    app.run(host="0.0.0.0", port=PORT)
