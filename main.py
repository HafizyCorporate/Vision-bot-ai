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

# --- 2. LOAD 4 OTAK AI SUPER ETLE ---
print("Memuat Otak 1: Deteksi Helm...")
model_helm = YOLO('helmet.pt') 

print("Memuat Otak 2: Deteksi Plat Nomor...")
model_plat = YOLO('license_plate_detector.pt') 

print("Memuat Otak 3: Pembaca Huruf (OCR)...")
reader = easyocr.Reader(['en'], gpu=False) 

print("Memuat Otak 4: Pelacak Motor (Anti Pejalan Kaki)...")
model_motor = YOLO('yolov8n.pt') # Otomatis download file 3MB dari Ultralytics

# --- FUNGSI TUKANG GAMBAR HELM, PLAT & ANTI PEJALAN KAKI ---
def gambar_semua(frame, hasil_helm, hasil_plat, hasil_motor):
    frame_gambar = frame.copy()
    jumlah_pelanggar = 0
    max_area_pelanggar = 0
    
    # 1. Simpan Koordinat Semua Motor yang Lewat
    daftar_motor = []
    for box in hasil_motor[0].boxes:
        mx1, my1, mx2, my2 = map(int, box.xyxy[0])
        daftar_motor.append((mx1, my1, mx2, my2))
        # Gambar kotak tipis warna putih buat menandai bodi motor
        cv2.rectangle(frame_gambar, (mx1, my1), (mx2, my2), (255, 255, 255), 1)
    
    # 2. Gambar Semua Plat Nomor di Video (Kotak Biru)
    for box in hasil_plat[0].boxes:
        px1, py1, px2, py2 = map(int, box.xyxy[0])
        cv2.rectangle(frame_gambar, (px1, py1), (px2, py2), (255, 0, 0), 2)
        cv2.putText(frame_gambar, "PLAT", (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    # 3. Gambar Helm (Sambil Mengecek Nempel Motor atau Nggak)
    for box in hasil_helm[0].boxes:
        hx1, hy1, hx2, hy2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        kelas_id = int(box.cls[0])
        nama_objek = model_helm.names[kelas_id].lower()
        
        # LOGIKA ANTI PEJALAN KAKI (Cek Irisan Kotak)
        nempel_motor = False
        for (mx1, my1, mx2, my2) in daftar_motor:
            # Jika kotak kepala overlap/bersentuhan dengan kotak motor
            if not (hx2 < mx1 or hx1 > mx2 or hy2 < my1 or hy1 > my2):
                nempel_motor = True
                break
        
        # PENGKONDISIAN WARNA & TILANG
        if "no" in nama_objek or "without" in nama_objek or "bare" in nama_objek:
            if nempel_motor:
                warna = (0, 0, 255) # MERAH (Joki Motor No Helm)
                label = f"NO HELM {conf:.2f}"
                jumlah_pelanggar += 1
                area = (hx2 - hx1) * (hy2 - hy1)
                if area > max_area_pelanggar: 
                    max_area_pelanggar = area
            else:
                warna = (128, 128, 128) # ABU-ABU (Pejalan Kaki No Helm)
                label = "PEJALAN KAKI"
        else:
            if nempel_motor:
                warna = (0, 255, 0) # HIJAU (Joki Motor Pakai Helm)
                label = f"HELM {conf:.2f}"
            else:
                warna = (128, 128, 128) # ABU-ABU (Pejalan Kaki Pakai Topi/Helm)
                label = "PEJALAN KAKI"
            
        cv2.rectangle(frame_gambar, (hx1, hy1), (hx2, hy2), warna, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_gambar, (hx1, hy1 - 20), (hx1 + tw, hy1), warna, -1)
        cv2.putText(frame_gambar, label, (hx1, hy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return frame_gambar, jumlah_pelanggar, max_area_pelanggar

# --- 3. FITUR UTAMA: RENDER VIDEO VIA TELEGRAM ---
@bot.message_handler(content_types=['video', 'document'])
def handle_video(message):
    bot.reply_to(message, "⚙️ [SYSTEM] Memproses Video (Resolusi HD)... AI melacak Motor, Helm, Plat, dan menyaring Pejalan Kaki!")
    
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
        
        largest_violator_ever = 0
        best_evidence_frame = None 
        best_clean_frame = None
        waktu_kejadian_ms = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            waktu_saat_ini_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                
            # Scan Motor (Class 3), Helm & Plat dengan resolusi besar
            results_motor = model_motor(frame, conf=0.15, classes=[3], imgsz=1280, verbose=False)
            results_helm = model_helm(frame, conf=0.05, imgsz=1920, verbose=False)
            results_plat = model_plat(frame, conf=0.05, imgsz=1920, verbose=False)
            
            frame_plotted, pelanggar_di_frame, area_pelanggar = gambar_semua(frame, results_helm, results_plat, results_motor)
            
            # Kunci momen saat PELANGGAR MOTOR paling deket sama kamera
            if area_pelanggar > largest_violator_ever:
                largest_violator_ever = area_pelanggar
                best_evidence_frame = frame_plotted.copy()
                best_clean_frame = frame.copy() 
                waktu_kejadian_ms = waktu_saat_ini_ms
                
            out.write(frame_plotted)
            
        cap.release()
        out.release()
        
        # --- 1. KIRIM VIDEO YANG SUDAH DIGAMBAR KOTAK ---
        with open(output_path, 'rb') as video_file:
            bot.send_video(message.chat.id, video_file, caption="🎥 *REKAMAN SELESAI*\nAI mengabaikan Pejalan Kaki (Kotak Abu-abu).", parse_mode='Markdown')
            
        # --- 2. JIKA ADA PELANGGAR, KIRIM SCREENSHOT + OCR ---
        if best_evidence_frame is not None and largest_violator_ever > 0:
            bot.send_message(message.chat.id, "🚨 *Pelanggar Ditemukan!* Mengekstrak Plat Nomor target...", parse_mode='Markdown')
            
            hasil_plat = model_plat(best_clean_frame, conf=0.05, imgsz=1920, verbose=False)
            
            plat_pelanggar = "Tidak Terbaca / Buram"
            max_plat_area = 0
            box_plat_terbaik = None
            
            if len(hasil_plat[0].boxes) > 0:
                for box in hasil_plat[0].boxes:
                    px1, py1, px2, py2 = map(int, box.xyxy[0])
                    p_area = (px2 - px1) * (py2 - py1)
                    if p_area > max_plat_area:
                        max_plat_area = p_area
                        box_plat_terbaik = (px1, py1, px2, py2)
                        
                if box_plat_terbaik is not None:
                    px1, py1, px2, py2 = box_plat_terbaik
                    potongan_plat = best_clean_frame[py1:py2, px1:px2]
                    
                    # Zoom natural buat dibaca OCR
                    plat_zoom = cv2.resize(potongan_plat, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                    teks_hasil = reader.readtext(plat_zoom, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                    
                    if teks_hasil:
                        plat_pelanggar = "".join(teks_hasil).replace(" ", "")
                        
                    cv2.rectangle(best_evidence_frame, (px1, py1), (px2, py2), (0, 255, 255), 4)
                    cv2.putText(best_evidence_frame, "TARGET PLAT", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
            detik_total = int(waktu_kejadian_ms / 1000)
            menit = detik_total // 60
            detik = detik_total % 60
            waktu_str = f"Menit {menit:02d} Detik {detik:02d}"
            
            bukti_path = "bukti_tilang.jpg"
            cv2.imwrite(bukti_path, best_evidence_frame)
            
            # --- FORMAT LAPORAN RAPI ---
            surat_tilang = (
                f"No plat : `{plat_pelanggar}`\n"
                f"Pelanggarannya : Tidak Menggunakan Helm\n"
                f"waktu : {waktu_str}"
            )
            
            with open(bukti_path, 'rb') as foto_bukti:
                bot.send_photo(message.chat.id, foto_bukti, caption=surat_tilang, parse_mode='Markdown')
            os.remove(bukti_path)
            
        else:
            bot.send_message(message.chat.id, "✅ *AMAN:* Tidak ada pengendara motor tanpa helm di video ini.", parse_mode='Markdown')
            
        os.remove(input_path)
        os.remove(output_path)
        
    except Exception as e:
        bot.reply_to(message, f"❌ [ERROR] {e}")

@app.route('/vision', methods=['POST'])
def vision_endpoint(): return jsonify({"status": "Web dinonaktifkan"}), 200

@bot.message_handler(commands=['start', 'land'])
def command_land(message): bot.reply_to(message, "🔴 ETLE ANTI-PEJALAN KAKI AKTIF!")

def run_bot(): bot.infinity_polling()

if __name__ == '__main__':
    threading.Thread(target=run_bot).start()
    app.run(host="0.0.0.0", port=PORT)
