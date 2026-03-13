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

print("Memuat Otak 4: Pelacak Kendaraan (Anti Pejalan Kaki & Anti Timestamp)...")
model_kendaraan = YOLO('yolov8n.pt')

# --- FUNGSI TUKANG GAMBAR LOGIKA TINGKAT TINGGI ---
def gambar_semua(frame, hasil_helm, hasil_plat, hasil_kendaraan, prev_motors):
    frame_gambar = frame.copy()
    jumlah_pelanggar = 0
    max_area_pelanggar = 0
    
    semua_kendaraan = [] # Buat filter plat nomor
    motor_berjalan = []
    motor_berhenti = []
    pusat_motor_baru = [] # Buat sensor gerak frame berikutnya
    
    # 1. ANALISIS KENDARAAN & SENSOR GERAK
    for box in hasil_kendaraan[0].boxes:
        kx1, ky1, kx2, ky2 = map(int, box.xyxy[0])
        kelas_id = int(box.cls[0])
        
        # Simpan semua kendaraan (Mobil, Motor, Bus, Truk) buat validasi plat
        if kelas_id in [2, 3, 5, 7]: 
            semua_kendaraan.append((kx1, ky1, kx2, ky2))
            
        # KHUSUS MOTOR (Class 3): Cek apakah dia jalan atau berhenti
        if kelas_id == 3:
            cx, cy = (kx1 + kx2) // 2, (ky1 + ky2) // 2
            pusat_motor_baru.append((cx, cy))
            
            is_moving = True
            if prev_motors:
                # Hitung jarak dengan motor di frame sebelumnya
                distances = [np.sqrt((cx - px)**2 + (cy - py)**2) for px, py in prev_motors]
                if min(distances) < 2.0: # Jika bergerak kurang dari 2 pixel = Berhenti
                    is_moving = False
                    
            if is_moving:
                motor_berjalan.append((kx1, ky1, kx2, ky2))
            else:
                motor_berhenti.append((kx1, ky1, kx2, ky2))

    # 2. VALIDASI PLAT NOMOR (Haram ngebaca tanggal CCTV)
    kotak_plat_valid = []
    for box in hasil_plat[0].boxes:
        px1, py1, px2, py2 = map(int, box.xyxy[0])
        pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
        
        valid = False
        for (vx1, vy1, vx2, vy2) in semua_kendaraan:
            # Plat harus berada di dalam / dekat bodi kendaraan
            if vx1 - 50 < pcx < vx2 + 50 and vy1 - 50 < pcy < vy2 + 50:
                valid = True
                break
                
        if valid:
            kotak_plat_valid.append((px1, py1, px2, py2))
            cv2.rectangle(frame_gambar, (px1, py1), (px2, py2), (255, 0, 0), 2)
            cv2.putText(frame_gambar, "PLAT", (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 3. VALIDASI HELM, PEJALAN KAKI, DAN STATUS BERHENTI
    for box in hasil_helm[0].boxes:
        hx1, hy1, hx2, hy2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        kelas_id = int(box.cls[0])
        nama_objek = model_helm.names[kelas_id].lower()
        
        hcx, hcy = (hx1 + hx2) // 2, (hy1 + hy2) // 2
        
        status_pengendara = "PEJALAN_KAKI"
        
        # Cek apakah nempel sama motor jalan? (Padding atas 100px biar kepala nggak dianggep melayang)
        for (mx1, my1, mx2, my2) in motor_berjalan:
            if mx1 - 50 < hcx < mx2 + 50 and my1 - 100 < hcy < my2 + 50:
                status_pengendara = "JALAN"
                break
                
        # Kalau bukan jalan, cek apakah nempel motor berhenti?
        if status_pengendara == "PEJALAN_KAKI":
            for (mx1, my1, mx2, my2) in motor_berhenti:
                if mx1 - 50 < hcx < mx2 + 50 and my1 - 100 < hcy < my2 + 50:
                    status_pengendara = "BERHENTI"
                    break

        # LOGIKA TILANG
        if status_pengendara == "PEJALAN_KAKI":
            warna = (128, 128, 128)
            label = "PEJALAN KAKI"
        elif status_pengendara == "BERHENTI":
            warna = (0, 255, 255) # KUNING
            label = "BERHENTI (AMAN)"
        elif status_pengendara == "JALAN":
            if "no" in nama_objek or "without" in nama_objek or "bare" in nama_objek:
                warna = (0, 0, 255) # MERAH
                label = f"NO HELM {conf:.2f}"
                jumlah_pelanggar += 1
                area = (hx2 - hx1) * (hy2 - hy1)
                if area > max_area_pelanggar: 
                    max_area_pelanggar = area
            else:
                warna = (0, 255, 0) # HIJAU
                label = f"HELM {conf:.2f}"
                
        cv2.rectangle(frame_gambar, (hx1, hy1), (hx2, hy2), warna, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_gambar, (hx1, hy1 - 20), (hx1 + tw, hy1), warna, -1)
        cv2.putText(frame_gambar, label, (hx1, hy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return frame_gambar, jumlah_pelanggar, max_area_pelanggar, pusat_motor_baru, kotak_plat_valid

# --- 3. FITUR UTAMA: RENDER VIDEO VIA TELEGRAM ---
@bot.message_handler(content_types=['video', 'document'])
def handle_video(message):
    bot.reply_to(message, "⚙️ [SYSTEM] Memproses Video... Mengaktifkan Sensor Gerak & Validasi Plat Anti-Timestamp!")
    
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
        best_plat_boxes = []
        waktu_kejadian_ms = 0
        
        prev_motors = [] # Variabel penyimpan sensor gerak
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            waktu_saat_ini_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                
            results_kendaraan = model_kendaraan(frame, conf=0.15, classes=[2, 3, 5, 7], imgsz=1280, verbose=False)
            results_helm = model_helm(frame, conf=0.05, imgsz=1920, verbose=False)
            results_plat = model_plat(frame, conf=0.05, imgsz=1920, verbose=False)
            
            frame_plotted, pelanggar_jalan, area_pelanggar, prev_motors, plat_valid = gambar_semua(
                frame, results_helm, results_plat, results_kendaraan, prev_motors
            )
            
            # Kunci Screenshot HANYA SAAT PELANGGAR BERJALAN & PALING JELAS
            if area_pelanggar > largest_violator_ever:
                largest_violator_ever = area_pelanggar
                best_evidence_frame = frame_plotted.copy()
                best_clean_frame = frame.copy() 
                best_plat_boxes = plat_valid
                waktu_kejadian_ms = waktu_saat_ini_ms
                
            out.write(frame_plotted)
            
        cap.release()
        out.release()
        
        # --- 1. KIRIM VIDEO ---
        with open(output_path, 'rb') as video_file:
            bot.send_video(message.chat.id, video_file, caption="🎥 *REKAMAN SELESAI*\nAI Pintar: Mengabaikan Pejalan Kaki & Motor Berhenti.", parse_mode='Markdown')
            
        # --- 2. JIKA ADA PELANGGAR, KIRIM SCREENSHOT + OCR ---
        if best_evidence_frame is not None and largest_violator_ever > 0:
            bot.send_message(message.chat.id, "🚨 *Pelanggar Ditemukan!* Mengekstrak Plat Nomor target...", parse_mode='Markdown')
            
            plat_pelanggar = "Tidak Terbaca / Buram"
            max_plat_area = 0
            box_plat_terbaik = None
            
            # Ambil plat dari kotak plat valid (bebas dari tulisan tanggal CCTV)
            if len(best_plat_boxes) > 0:
                for (px1, py1, px2, py2) in best_plat_boxes:
                    p_area = (px2 - px1) * (py2 - py1)
                    if p_area > max_plat_area:
                        max_plat_area = p_area
                        box_plat_terbaik = (px1, py1, px2, py2)
                        
                if box_plat_terbaik is not None:
                    px1, py1, px2, py2 = box_plat_terbaik
                    potongan_plat = best_clean_frame[py1:py2, px1:px2]
                    
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
            
            # FORMAT LAPORAN MINTAAN BOS
            surat_tilang = (
                f"No plat : `{plat_pelanggar}`\n"
                f"Pelanggarannya : Tidak Menggunakan Helm\n"
                f"waktu : {waktu_str}"
            )
            
            with open(bukti_path, 'rb') as foto_bukti:
                bot.send_photo(message.chat.id, foto_bukti, caption=surat_tilang, parse_mode='Markdown')
            os.remove(bukti_path)
            
        else:
            bot.send_message(message.chat.id, "✅ *AMAN:* Tidak ada pengendara motor melanggar yang bergerak/jalan di video ini.", parse_mode='Markdown')
            
        os.remove(input_path)
        os.remove(output_path)
        
    except Exception as e:
        bot.reply_to(message, f"❌ [ERROR] {e}")

@app.route('/vision', methods=['POST'])
def vision_endpoint(): return jsonify({"status": "Web dinonaktifkan"}), 200

@bot.message_handler(commands=['start', 'land'])
def command_land(message): bot.reply_to(message, "🔴 ETLE PINTAR (ANTI-BUG) AKTIF!")

def run_bot(): bot.infinity_polling()

if __name__ == '__main__':
    threading.Thread(target=run_bot).start()
    app.run(host="0.0.0.0", port=PORT)
