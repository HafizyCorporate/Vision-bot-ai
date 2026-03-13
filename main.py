import os
import base64
import cv2
import numpy as np
import telebot
import threading
import easyocr # <-- KITA PANGGIL LAGI OTAK PEMBACA HURUFNYA
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from ultralytics import YOLO

# --- 1. SETUP VARIABEL ---
TOKEN = os.environ.get("BOT_TOKEN", "TOKEN_BOT_KAMU")
CHAT_ID = os.environ.get("CHAT_ID", "CHAT_ID_KAMU_BERUPA_ANGKA")
PORT = int(os.environ.get("PORT", 8080))

bot = telebot.TeleBot(TOKEN)
app = Flask(__name__)

# --- 2. LOAD 2 OTAK AI SEKALIGUS ---
print("Memuat model ETLE Khusus Helm...")
model = YOLO('helmet.pt') 

print("Memuat model OCR Pembaca Plat...")
# GPU=False agar aman di server gratisan/standar
reader = easyocr.Reader(['en'], gpu=False) 

# --- FUNGSI TUKANG GAMBAR KOTAK CUSTOM ---
def gambar_custom_kotak(frame, hasil_ai):
    frame_gambar = frame.copy()
    jumlah_pelanggar = 0
    max_area_pelanggar = 0 # Buat nyari momen paling dekat kamera
    
    for box in hasil_ai[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        kelas_id = int(box.cls[0])
        nama_objek = model.names[kelas_id].lower()
        
        if "no" in nama_objek or "without" in nama_objek or "bare" in nama_objek:
            warna = (0, 0, 255) # MERAH
            label = f"TIDAK PAKAI HELM {conf:.2f}"
            jumlah_pelanggar += 1
            
            # Hitung seberapa besar pelanggar di layar (buat SS terbaik)
            area = (x2 - x1) * (y2 - y1)
            if area > max_area_pelanggar:
                max_area_pelanggar = area
                
        else:
            warna = (0, 255, 0) # HIJAU
            label = f"PAKAI HELM {conf:.2f}"
            
        cv2.rectangle(frame_gambar, (x1, y1), (x2, y2), warna, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_gambar, (x1, y1 - 20), (x1 + tw, y1), warna, -1)
        cv2.putText(frame_gambar, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return frame_gambar, jumlah_pelanggar, max_area_pelanggar

# --- 3. FITUR RENDER VIDEO VIA TELEGRAM ---
@bot.message_handler(content_types=['video', 'document'])
def handle_video(message):
    bot.reply_to(message, "⚙️ [SYSTEM] Video ETLE diterima! Memproses scanning helm dan plat nomor...")
    
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
        
        max_pelanggar_terekam = 0 
        largest_violator_ever = 0
        best_evidence_frame = None # Tempat nyimpan Screenshot Pelanggar
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model(frame, conf=0.35, verbose=False)
            frame_plotted, pelanggar_di_frame, max_area = gambar_custom_kotak(frame, results)
            
            if pelanggar_di_frame > max_pelanggar_terekam:
                max_pelanggar_terekam = pelanggar_di_frame
                
            if max_area > largest_violator_ever:
                largest_violator_ever = max_area
                best_evidence_frame = frame_plotted.copy()
                
            out.write(frame_plotted)
            
        cap.release()
        out.release()
        
        # --- LANGKAH 1: KIRIM BALIK VIDEO RENDER ---
        with open(output_path, 'rb') as video_file:
            bot.send_video(message.chat.id, video_file, caption="🎥 *REKAMAN ETLE SELESAI*\n🔴 Merah: Pelanggar | 🟢 Hijau: Aman", parse_mode='Markdown')
            
        # --- LANGKAH 2: EKSEKUSI SS & OCR PLAT NOMOR ---
        if best_evidence_frame is not None:
            bot.send_message(message.chat.id, "🔍 *Mengekstrak Plat Nomor* dari foto barang bukti...", parse_mode='Markdown')
            
            # Menjalankan Otak OCR KHUSUS di 1 foto ini saja (Aman & Cepat!)
            teks_hasil = reader.readtext(best_evidence_frame, detail=0)
            
            # Filter teks: gabungkan yang dibaca OCR, kalau kosong tulis 'Buram'
            if teks_hasil:
                plat_terbaca = " ".join(teks_hasil)
            else:
                plat_terbaca = "Tidak terdeteksi / Buram"

            # Simpan SS ke file
            bukti_path = "bukti_tilang.jpg"
            cv2.imwrite(bukti_path, best_evidence_frame)
            
            # Buat Surat Tilang
            waktu_wib = datetime.utcnow() + timedelta(hours=7)
            surat_tilang = (
                "🚨 *SURAT TILANG ELEKTRONIK (ETLE)* 🚨\n\n"
                f"📅 *Tanggal:* {waktu_wib.strftime('%d %B %Y')}\n"
                f"⏰ *Waktu:* {waktu_wib.strftime('%H:%M:%S WIB')}\n"
                f"📍 *Lokasi:* Kamera Mobile Telegram\n"
                f"⚠️ *Jenis Pelanggaran:* Tidak Menggunakan Helm SNI\n"
                f"👤 *Terduga Pelanggar:* {max_pelanggar_terekam} Orang\n\n"
                f"🔍 *HASIL SCAN PLAT NOMOR:*\n"
                f"👉 `{plat_terbaca}` 👈\n\n"
                "Status: Menunggu Validasi Petugas."
            )
            
            # Kirim SS + Catatan
            with open(bukti_path, 'rb') as foto_bukti:
                bot.send_photo(message.chat.id, foto_bukti, caption=surat_tilang, parse_mode='Markdown')
            os.remove(bukti_path)
            
        else:
            bot.send_message(message.chat.id, "✅ *LAPORAN AMAN:* Tidak ditemukan pengendara tanpa helm di video ini.", parse_mode='Markdown')
            
        # Bersihkan sampah video
        os.remove(input_path)
        os.remove(output_path)
        
    except Exception as e:
        bot.reply_to(message, f"❌ [ERROR] Gagal memproses: {e}")
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
        frame_plotted, jumlah_pelanggar, _ = gambar_custom_kotak(img, results)
        
        _, buffer = cv2.imencode('.jpg', frame_plotted)
        foto_final = buffer.tobytes()

        if jumlah_pelanggar > 0:
            waktu_wib = datetime.utcnow() + timedelta(hours=7)
            surat_tilang = f"🚨 *ETLE ALERT (FOTO)* 🚨\n\nPelanggar: {jumlah_pelanggar} Orang\nWaktu: {waktu_wib.strftime('%H:%M:%S WIB')}"
            bot.send_photo(CHAT_ID, foto_final, caption=surat_tilang, parse_mode='Markdown')
            
        return jsonify({"status": "Diproses"}), 200
            
    except Exception as e:
        return jsonify({"error": "Server error"}), 500

# --- 5. FUNGSI TELEGRAM & RUNNER ---
@bot.message_handler(commands=['start', 'land'])
def command_land(message):
    bot.reply_to(message, "🔴 ETLE SIAGA!\n\nKirimkan VIDEO mentahan. Sistem akan:\n1. Merender deteksi helm.\n2. Mengirim balik video.\n3. Mengambil SS Pelanggar.\n4. Membaca Plat Nomor (OCR).")

def run_bot():
    bot.infinity_polling()

if __name__ == '__main__':
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.start()
    print(f"🔥 Server ETLE aktif di port {PORT}...")
    app.run(host="0.0.0.0", port=PORT)
