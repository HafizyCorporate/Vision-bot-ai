import os
import base64
import cv2
import numpy as np
import telebot
import threading
import easyocr
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from ultralytics import YOLO

# --- 1. SETUP VARIABEL ---
TOKEN = os.environ.get("BOT_TOKEN", "TOKEN_BOT_KAMU")
CHAT_ID = os.environ.get("CHAT_ID", "CHAT_ID_KAMU_BERUPA_ANGKA")
PORT = int(os.environ.get("PORT", 8080))

bot = telebot.TeleBot(TOKEN)
app = Flask(__name__)

# --- 2. LOAD 3 OTAK AI SUPER ETLE ---
print("Memuat Otak 1: Deteksi Helm...")
model_helm = YOLO('helmet.pt') 

print("Memuat Otak 2: Deteksi Plat Nomor...")
model_plat = YOLO('license_plate_detector.pt') 

print("Memuat Otak 3: Pembaca Huruf (OCR)...")
reader = easyocr.Reader(['en'], gpu=False) 

# --- FUNGSI TUKANG GAMBAR + LOGIKA GARIS MIRING ---
def gambar_custom_kotak(frame, hasil_ai):
    frame_gambar = frame.copy()
    h, w = frame_gambar.shape[:2]
    
    # 🚨 SETTING GARIS MIRING (BISA KAMU UBAH SESUAI CCTV) 🚨
    # pt1 = Titik Kiri (x=0, y=50% dari layar)
    # pt2 = Titik Kanan (x=mentok kanan, y=80% dari layar)
    # Efeknya: Garis akan menukik miring dari kiri-atas ke kanan-bawah.
    pt1 = (0, int(h * 0.5))       
    pt2 = (w, int(h * 0.8))       
    
    cv2.line(frame_gambar, pt1, pt2, (0, 165, 255), 3) # Gambar Garis Oranye
    cv2.putText(frame_gambar, "GARIS STOP MIRING", (10, pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    jumlah_pelanggar_helm = 0
    jumlah_pelanggar_garis = 0
    max_area_pelanggar = 0 
    
    for box in hasil_ai[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        kelas_id = int(box.cls[0])
        nama_objek = model_helm.names[kelas_id].lower()
        
        # Cari titik pusat ban (Tengah X, Bawah Y)
        titik_tengah_x = int((x1 + x2) / 2)
        titik_bawah_y = y2
        
        # RUMUS MATEMATIKA GARIS MIRING: Mencari batas Y tepat di bawah ban motor
        if pt2[0] != pt1[0]:
            kemiringan_m = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
            batas_y_di_titik_x = pt1[1] + kemiringan_m * (titik_tengah_x - pt1[0])
        else:
            batas_y_di_titik_x = pt1[1]
            
        # Gambar titik kuning di posisi "Ban Motor" biar kelihatan AI ngelacak apa
        cv2.circle(frame_gambar, (titik_tengah_x, titik_bawah_y), 5, (0, 255, 255), -1)

        # LOGIKA 1: CEK HELM
        if "no" in nama_objek or "without" in nama_objek or "bare" in nama_objek:
            warna = (0, 0, 255) # MERAH
            label = f"NO HELM {conf:.2f}"
            jumlah_pelanggar_helm += 1
            area = (x2 - x1) * (y2 - y1)
            if area > max_area_pelanggar: max_area_pelanggar = area
        else:
            warna = (0, 255, 0) # HIJAU
            label = f"HELM {conf:.2f}"
            
        # LOGIKA 2: CEK TEROBOS GARIS MIRING
        # Kalau Ban Motor (titik_bawah_y) melewati batas kemiringan
        if titik_bawah_y > batas_y_di_titik_x:
            warna = (255, 0, 255) # UNGU (Pelanggaran Marka)
            label = f"TEROBOS BATAS!"
            jumlah_pelanggar_garis += 1
            area = (x2 - x1) * (y2 - y1)
            if area > max_area_pelanggar: max_area_pelanggar = area 
                
        # GAMBAR KOTAKNYA
        cv2.rectangle(frame_gambar, (x1, y1), (x2, y2), warna, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_gambar, (x1, y1 - 20), (x1 + tw, y1), warna, -1)
        cv2.putText(frame_gambar, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return frame_gambar, jumlah_pelanggar_helm, jumlah_pelanggar_garis, max_area_pelanggar

# --- 3. FITUR UTAMA: RENDER VIDEO VIA TELEGRAM ---
@bot.message_handler(content_types=['video', 'document'])
def handle_video(message):
    bot.reply_to(message, "⚙️ [SYSTEM] Video diterima! Mengaktifkan Radar Garis Miring...")
    
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
        
        max_helm = 0 
        max_garis = 0
        largest_violator_ever = 0
        best_evidence_frame = None 
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            results = model_helm(frame, conf=0.20, imgsz=640, verbose=False)
            frame_plotted, pel_helm, pel_garis, max_area = gambar_custom_kotak(frame, results)
            
            if pel_helm > max_helm: max_helm = pel_helm
            if pel_garis > max_garis: max_garis = pel_garis
                
            if max_area > largest_violator_ever:
                largest_violator_ever = max_area
                best_evidence_frame = frame_plotted.copy()
                
            out.write(frame_plotted)
            
        cap.release()
        out.release()
        
        # --- KIRIM VIDEO ---
        with open(output_path, 'rb') as video_file:
            bot.send_video(message.chat.id, video_file, caption="🎥 *REKAMAN SELESAI*\n🟢 Hijau: Aman\n🔴 Merah: No Helm\n🟣 Ungu: Terobos Garis Miring", parse_mode='Markdown')
            
        # --- KIRIM SS & OCR ---
        if best_evidence_frame is not None and (max_helm > 0 or max_garis > 0):
            bot.send_message(message.chat.id, "🔍 *Mencari Plat Nomor pelanggar...*", parse_mode='Markdown')
            
            hasil_plat = model_plat(best_evidence_frame, conf=0.15, verbose=False)
            plat_terbaca = "Plat tidak terlihat di kamera"
            
            if len(hasil_plat[0].boxes) > 0:
                box_plat = hasil_plat[0].boxes[0]
                px1, py1, px2, py2 = map(int, box_plat.xyxy[0])
                potongan_plat = best_evidence_frame[py1:py2, px1:px2]
                teks_hasil = reader.readtext(potongan_plat, detail=0, mag_ratio=2.5)
                
                if teks_hasil: plat_terbaca = " ".join(teks_hasil).upper()
                else: plat_terbaca = "Terdeteksi Plat, huruf buram"
                    
                cv2.rectangle(best_evidence_frame, (px1, py1), (px2, py2), (255, 0, 0), 3) 
                cv2.putText(best_evidence_frame, "PLAT", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            bukti_path = "bukti_tilang.jpg"
            cv2.imwrite(bukti_path, best_evidence_frame)
            
            jenis_pelanggaran = []
            if max_helm > 0: jenis_pelanggaran.append("Tanpa Helm")
            if max_garis > 0: jenis_pelanggaran.append("Terobos Garis CCTV Miring")
            teks_pelanggaran = " & ".join(jenis_pelanggaran)

            waktu_wib = datetime.utcnow() + timedelta(hours=7)
            surat_tilang = (
                "🚨 *SURAT TILANG ETLE* 🚨\n\n"
                f"📅 *Tanggal:* {waktu_wib.strftime('%d %B %Y')}\n"
                f"⚠️ *Pelanggaran:* {teks_pelanggaran}\n"
                f"🔍 *PLAT NOMOR:* 👉 `{plat_terbaca}` 👈"
            )
            
            with open(bukti_path, 'rb') as foto_bukti:
                bot.send_photo(message.chat.id, foto_bukti, caption=surat_tilang, parse_mode='Markdown')
            os.remove(bukti_path)
            
        else:
            bot.send_message(message.chat.id, "✅ *AMAN:* Tidak ada pelanggaran.", parse_mode='Markdown')
            
        os.remove(input_path)
        os.remove(output_path)
        
    except Exception as e:
        bot.reply_to(message, f"❌ [ERROR] {e}")

@app.route('/vision', methods=['POST'])
def vision_endpoint(): return jsonify({"status": "Web dinonaktifkan"}), 200
@bot.message_handler(commands=['start', 'land'])
def command_land(message): bot.reply_to(message, "🔴 ETLE CCTV MIRING AKTIF!")
def run_bot(): bot.infinity_polling()
if __name__ == '__main__':
    threading.Thread(target=run_bot).start()
    app.run(host="0.0.0.0", port=PORT)
