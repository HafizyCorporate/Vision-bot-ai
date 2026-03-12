import os
import cv2
import telebot
import numpy as np
import requests
from ultralytics import YOLO

# Ambil token dari server Railway nanti
BOT_TOKEN = os.environ.get('BOT_TOKEN')
bot = telebot.TeleBot(BOT_TOKEN)

print("Memuat model AI Vision (YOLOv8)...")
model = YOLO('yolov8n.pt')
print("Sistem Siap!")

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "🔥 Sistem AI Aktif. Kirimkan foto ke sini, dan saya akan mendeteksi objek di dalamnya.")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        bot.reply_to(message, "Menganalisis gambar... ⏳")
        
        # Download foto dari Telegram
        file_info = bot.get_file(message.photo[-1].file_id)
        file_url = f'https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}'
        response = requests.get(file_url)
        
        # Ubah foto agar bisa dibaca AI
        image_array = np.asarray(bytearray(response.content), dtype="uint8")
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        # AI mulai mendeteksi
        results = model(img)
        annotated_frame = results[0].plot()
        
        # Catat objek apa saja yang ketemu
        detected_items = [model.names[int(box.cls[0])] for r in results for box in r.boxes]
        
        if len(detected_items) > 0:
            summary = {item: detected_items.count(item) for item in set(detected_items)}
            teks_balasan = "🎯 **Hasil Deteksi:**\n"
            for obj, count in summary.items():
                teks_balasan += f"- {obj}: {count}\n"
        else:
            teks_balasan = "⚠️ Tidak ada objek yang dikenali."
            
        # Simpan & kirim balik foto hasilnya
        cv2.imwrite("hasil.jpg", annotated_frame)
        with open("hasil.jpg", 'rb') as photo:
            bot.send_photo(message.chat.id, photo, caption=teks_balasan)
            
    except Exception as e:
        bot.reply_to(message, "Error: Gambar gagal diproses.")

if __name__ == '__main__':
    bot.infinity_polling(timeout=10, long_polling_timeout=5)
