import os
import time
import uuid
import asyncio
import gc
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from faster_whisper import WhisperModel

import numpy as np
import librosa

# ============================================================
# НАСТРОЙКИ
# ============================================================
MAX_CONCURRENT_TRANSCRIBES = 2
cpu_threads = 3

print("Загрузка модели base...")
model = WhisperModel(
    "base", device="cpu", compute_type="int8", cpu_threads=cpu_threads, num_workers=1,
)
print("Модель готова!")

transcribe_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIBES)
queue_size = 0
queue_lock = asyncio.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    print("Завершение работы...")

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================================
# АНАЛИЗ БЕГЛОСТИ
# ============================================================
def analyze_fluency(segments, total_duration, pause_threshold=0.5):
    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append({"start": w.start, "end": w.end, "word": w.word})

    if not words or total_duration <= 0:
        return {"words_count": 0, "words_per_minute": 0, "pauses_count": 0, "pauses_total_s": 0, "avg_pause_s": 0}

    pauses = []
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap >= pause_threshold:
            pauses.append(gap)

    if words[0]["start"] >= pause_threshold: pauses.append(words[0]["start"])
    tail_gap = total_duration - words[-1]["end"]
    if tail_gap >= pause_threshold: pauses.append(tail_gap)

    words_count = len(words)
    wpm = words_count / (total_duration / 60.0)
    pauses_total = sum(pauses)
    avg_pause = pauses_total / len(pauses) if pauses else 0

    return {
        "words_count": words_count, "words_per_minute": round(wpm, 1),
        "pauses_count": len(pauses), "pauses_total_s": round(pauses_total, 2),
        "avg_pause_s": round(avg_pause, 2)
    }

# ============================================================
# АНАЛИЗ АМПЛИТУДЫ
# ============================================================
def analyze_audio_energy(file_path):
    y = None
    rms = None
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True, dtype=np.float32)
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
        total_duration = len(y) / sr
        
        num_points = 250
        step = max(1, len(rms) // num_points)
        
        rms_downsampled = []
        for i in range(0, len(rms), step):
            rms_downsampled.append({
                "time": round(float(times[i]), 2),
                "value": round(float(rms[i]), 3)
            })
            
        if times[-1] < total_duration - 0.5:
            rms_downsampled.append({
                "time": round(float(total_duration), 2),
                "value": round(float(rms[-1]), 3)
            })

        return rms_downsampled, times, rms
    except Exception as e:
        print(f"[ENERGY ERROR] {e}")
        return [], [], []
    finally:
        if y is not None: del y

# Маппинг слов на амплитуду
def map_words_to_energy(words_data, times, rms):
    if len(times) == 0 or len(rms) == 0:
        return words_data

    for w in words_data:
        mask = (times >= w["start"]) & (times <= w["end"])
        if np.any(mask):
            avg_word_energy = np.mean(rms[mask])
            w["amplitude"] = round(float(avg_word_energy), 3)
        else:
            w["amplitude"] = 0

    return words_data

def _sync_transcribe(save_path):
    segments, info = model.transcribe(
        save_path, language="en", beam_size=1, word_timestamps=True, vad_filter=False, condition_on_previous_text=False
    )
    return list(segments), info

def _sync_process_all(save_path):
    segments_list, info = _sync_transcribe(save_path)
    rms_data, times, rms_raw = analyze_audio_energy(save_path)
    
    # Собираем слова
    words_data = []
    for seg in segments_list:
        if seg.words:
            for w in seg.words:
                words_data.append({"word": w.word.strip(), "start": round(w.start, 2), "end": round(w.end, 2)})

    # Мапим амплитуду на слова
    words_data = map_words_to_energy(words_data, times, rms_raw)
    
    if rms_raw is not None: del rms_raw
    gc.collect()
    
    return segments_list, info, rms_data, words_data

@app.post("/transcribe")
async def transcribe(request: Request, file: UploadFile = File(...)):
    global queue_size
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Файл должен быть аудио")

    t_start = time.perf_counter()
    ext = os.path.splitext(file.filename)[1] or ".webm"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uuid.uuid4().hex[:6]}{ext}"
    save_path = os.path.join(UPLOAD_DIR, filename)

    size_bytes = 0
    with open(save_path, "wb") as f:
        while chunk := await file.read(65536):
            f.write(chunk); size_bytes += len(chunk)

    t_saved = time.perf_counter()
    save_time = t_saved - t_start

    if await request.is_disconnected():
        if os.path.exists(save_path): os.remove(save_path)
        raise HTTPException(status_code=499, detail="Клиент отключился")

    async with queue_lock:
        queue_size += 1
        position = queue_size

    try:
        async with transcribe_semaphore:
            async with queue_lock:
                queue_size -= 1

            if await request.is_disconnected():
                if os.path.exists(save_path): os.remove(save_path)
                raise HTTPException(status_code=499, detail="Клиент отключился")

            print(f"[START] {filename} | processing...")
            segments_list, info, rms_data, words_data = await run_in_threadpool(_sync_process_all, save_path)

            text = " ".join(s.text for s in segments_list).strip()
            fluency = analyze_fluency(segments_list, info.duration)

            t_done = time.perf_counter()
            total_time = t_done - t_start

            return {
                "text": text,
                "duration": round(info.duration, 2),
                "fluency": fluency,
                "words_data": words_data,
                "amplitude_data": rms_data, # Переименовали rms_data в amplitude_data
                "timings": {"save_ms": round(save_time * 1000), "transcribe_s": round(t_done - t_saved, 2), "total_s": round(total_time, 2)},
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")
    finally:
        gc.collect()