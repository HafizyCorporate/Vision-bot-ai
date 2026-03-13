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

def gambar_custom_kotak(frame, hasil_ai):
    frame_gambar = frame.copy()
    jumlah_pelanggar = 0
    max_area_rider = 0 
    
    for box in hasil_ai[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        kelas_id = int(box.cls[0])
        nama_objek = model_helm.names[kelas_id].lower()
        
        area = (x2 - x1) * (y2 - y1)
        if area > max_area_rider: 
            max_area_rider = area
            
        if "no" in nama_objek or "without" in nama_objek or "bare" in nama_objek:
            warna = (0, 0, 255) # MERAH
            label = f"NO HELM {conf:.2f}"
            jumlah_pelanggar += 1
        else:
            warna = (0, 255, 0) # HIJAU
            label = f"HELM {conf:.2f}"
            
        cv2.rectangle(frame_gambar, (x1, y1), (x2, y2), warna, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_gambar, (x1, y1 - 20), (x1 + tw, y1), warna, -1)
        cv2.putText(frame_gambar, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return frame_gambar, jumlah_pelanggar, max_area_rider

@bot.message_handler(content_types=['video', 'document'])
def handle_video(message):
    bot.reply_to(message, "⚙️ [SYSTEM] Memproses Video... Mengaktifkan Mode Natural & Perekam Waktu (Timestamp)!")
    
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
        largest_rider_ever = 0
        best_evidence_frame = None 
        waktu_kejadian_ms = 0 # Menyimpan waktu saat jepretan terjadi
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            waktu_saat_ini_ms = cap.get(cv2.CAP_PROP_POS_MSEC) # Ambil waktu video berjalan
                
            # Mode Full HD tetep diaktifin biar CCTV jauh bisa kebaca
            results = model_helm(frame, conf=0.05, imgsz=1920, verbose=False)
            frame_plotted, pelanggar_di_frame, max_area = gambar_custom_kotak(frame, results)
            
            if pelanggar_di_frame > max_pelanggar_terekam: 
                max_pelanggar_terekam = pelanggar_di_frame
                
            if max_area > largest_rider_ever:
                largest_rider_ever = max_area
                best_evidence_frame = frame_plotted.copy()
                waktu_kejadian_ms = waktu_saat_ini_ms # Kunci waktu kejadiannya!
                
            out.write(frame_plotted)
            
        cap.release()
        out.release()
        
        with open(output_path, 'rb') as video_file:
            bot.send_video(message.chat.id, video_file, caption="🎥 *REKAMAN SELESAI*\n🟢 Aman | 🔴 No Helm", parse_mode='Markdown')
            
        if best_evidence_frame is not None:
            bot.send_message(message.chat.id, "🔍 *Mengekstrak Plat Nomor (Natural Mode)...*", parse_mode='Markdown')
            
            hasil_plat = model_plat(best_evidence_frame, conf=0.05, imgsz=1920, verbose=False)
            daftar_plat_terbaca = []
            
            if len(hasil_plat[0].boxes) > 0:
                for i, box_plat in enumerate(hasil_plat[0].boxes):
                    px1, py1, px2, py2 = map(int, box_plat.xyxy[0])
                    potongan_plat = best_evidence_frame[py1:py2, px1:px2]
                    
                    # 🔥 TAKTIK NATURAL: Cuma di-zoom 3x tanpa mengubah warna sama sekali 🔥
                    plat_zoom = cv2.resize(potongan_plat, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                    
                    # Langsung baca pakai OCR
                    teks_hasil = reader.readtext(plat_zoom, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                    
                    if teks_hasil: 
                        teks_bersih = "".join(teks_hasil).replace(" ", "")
                        daftar_plat_terbaca.append(f"👉 `{teks_bersih}`")
                    else: 
                        daftar_plat_terbaca.append(f"👉 `[Plat {i+1} Tidak Terbaca]`")
                        
                    cv2.rectangle(best_evidence_frame, (px1, py1), (px2, py2), (255, 0, 0), 3) 
                    cv2.putText(best_evidence_frame, f"PLAT {i+1}", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                daftar_plat_terbaca.append("❌ Tidak ada plat nomor yang terdeteksi.")

            teks_semua_plat = "\n".join(daftar_plat_terbaca)
            bukti_path = "bukti_tilang.jpg"
            cv2.imwrite(bukti_path, best_evidence_frame)
            
            # --- KONVERSI WAKTU KEJADIAN ---
            detik_total = int(waktu_kejadian_ms / 1000)
            menit_kejadian = detik_total // 60
            detik_kejadian = detik_total % 60
            waktu_video_str = f"{menit_kejadian:02d}:{detik_kejadian:02d}"
            
            waktu_wib = datetime.utcnow() + timedelta(hours=7)
            status_helm = "Ada Pelanggaran (Tidak Pakai Helm)" if max_pelanggar_terekam > 0 else "Aman (Semua Pakai Helm)"
            
            surat_tilang = (
                "🚨 *BUKTI PELANGGARAN ETLE* 🚨\n\n"
                f"📅 *Tanggal:* {waktu_wib.strftime('%d %B %Y')}\n"
                f"⏰ *Waktu Server:* {waktu_wib.strftime('%H:%M:%S WIB')}\n"
                f"⏱️ *Waktu di Video:* Detik ke-{waktu_video_str}\n"
                f"⚠️ *Status:* {status_helm}\n\n"
                f"🔍 *HASIL SCAN PLAT NOMOR:*\n"
                f"{teks_semua_plat}"
            )
            
            with open(bukti_path, 'rb') as foto_bukti:
                bot.send_photo(message.chat.id, foto_bukti, caption=surat_tilang, parse_mode='Markdown')
            os.remove(bukti_path)
            
        else:
            bot.send_message(message.chat.id, "✅ *AMAN:* Tidak ada kendaraan di video ini.", parse_mode='Markdown')
            
        os.remove(input_path)
        os.remove(output_path)
        
    except Exception as e:
        bot.reply_to(message, f"❌ [ERROR] {e}")

@app.route('/vision', methods=['POST'])
def vision_endpoint(): return jsonify({"status": "Web dinonaktifkan"}), 200

@bot.message_handler(commands=['start', 'land'])
def command_land(message): bot.reply_to(message, "🔴 ETLE NATURAL MODE AKTIF!")

def run_bot(): bot.infinity_polling()

if __name__ == '__main__':
    threading.Thread(target=run_bot).start()
    app.run(host="0.0.0.0", port=PORT)
