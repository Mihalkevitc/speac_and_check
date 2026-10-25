import os
import time
import uuid
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from faster_whisper import WhisperModel

# ============================================================
# НАСТРОЙКИ ПРОИЗВОДИТЕЛЬНОСТИ (под CPU)
# ============================================================
# Сколько транскрибаций могут выполняться ОДНОВРЕМЕННО.
# Для CPU это должно быть НЕБОЛЬШИМ. 2 — безопасный старт для 8-ядерного CPU.
# Если поставить 3-4, каждая задача получит меньше ядер и станет медленнее.
MAX_CONCURRENT_TRANSCRIBES = 2

# Сколько потоков CPU использует ОДНА транскрибация.
# При MAX_CONCURRENT=2 и 8 ядрах: 2 × 3 = 6 потоков, остальное — ОС и FastAPI.
cpu_threads = 3

# ============================================================
# МОДЕЛЬ (загружается один раз при старте)
# ============================================================
print("Загрузка модели base...")
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
    cpu_threads=cpu_threads,
    num_workers=1,  # ВАЖНО: 1 воркер внутри модели, параллелизм — через семафор
)

# model = WhisperModel(
#     "nyrahealth/faster_CrisperWhisper", 
#     device="cpu", 
#     compute_type="int8",
#     cpu_threads=cpu_threads
# )

print("Модель готова!")

# Семафор: пропускает не более MAX_CONCURRENT_TRANSCRIBES задач одновременно
transcribe_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIBES)

# Счётчик очереди для информативности (сколько ждут)
queue_size = 0
queue_lock = asyncio.Lock()

# ============================================================
# LIFESPAN (опционально, для graceful shutdown)
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    print("Завершение работы, ожидание активных задач...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================================
# БЕГЛОСТЬ (без изменений, но с защитой от пустых сегментов)
# ============================================================
def analyze_fluency(segments, total_duration, pause_threshold=0.5):
    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append({"start": w.start, "end": w.end, "word": w.word})

    if not words or total_duration <= 0:
        return {
            "words_count": 0, "words_per_minute": 0,
            "pauses_count": 0, "pauses_total_s": 0, "avg_pause_s": 0,
        }

    pauses = []
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap >= pause_threshold:
            pauses.append(gap)

    if words[0]["start"] >= pause_threshold:
        pauses.append(words[0]["start"])
    tail_gap = total_duration - words[-1]["end"]
    if tail_gap >= pause_threshold:
        pauses.append(tail_gap)

    words_count = len(words)
    wpm = words_count / (total_duration / 60.0)
    pauses_total = sum(pauses)
    avg_pause = pauses_total / len(pauses) if pauses else 0

    return {
        "words_count": words_count,
        "words_per_minute": round(wpm, 1),
        "pauses_count": len(pauses),
        "pauses_total_s": round(pauses_total, 2),
        "avg_pause_s": round(avg_pause, 2),
    }

def _sync_transcribe(save_path):
    segments, info = model.transcribe(
        save_path,
        language="en",
        beam_size=1,
        word_timestamps=True,
        vad_filter=False,  # для CPU лучше False - VAD добавляет оверхед
    )
    return list(segments), info

# ============================================================
# ЭНДПОИНТ
# ============================================================
@app.post("/transcribe")
async def transcribe(request: Request, file: UploadFile = File(...)):
    global queue_size

    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Файл должен быть аудио")

    t_start = time.perf_counter()

    # --- Сохранение файла чанками (защита от OOM) ---
    ext = os.path.splitext(file.filename)[1] or ".webm"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uuid.uuid4().hex[:6]}{ext}"
    save_path = os.path.join(UPLOAD_DIR, filename)

    size_bytes = 0
    with open(save_path, "wb") as f:
        while chunk := await file.read(65536):
            f.write(chunk)
            size_bytes += len(chunk)

    size_kb = size_bytes / 1024
    t_saved = time.perf_counter()
    save_time = t_saved - t_start

    # --- Проверка: не отключился ли клиент, пока мы ждём в очереди ---
    if await request.is_disconnected():
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=499, detail="Клиент отключился")

    # --- Увеличиваем счётчик очереди (для логов) ---
    async with queue_lock:
        queue_size += 1
        position = queue_size

    print(f"[QUEUE] {filename} | size={size_kb:.0f}KB | position={position} | wait...")

    try:
        # --- Ждём семафор: здесь запрос стоит в очереди, не нагружая CPU ---
        async with transcribe_semaphore:
            async with queue_lock:
                queue_size -= 1

            # Ещё раз проверяем, не отключился ли клиент, пока ждал
            if await request.is_disconnected():
                if os.path.exists(save_path):
                    os.remove(save_path)
                raise HTTPException(status_code=499, detail="Клиент отключился")

            print(f"[START] {filename} | waited in queue, now processing")

            segments_list, info = await run_in_threadpool(_sync_transcribe, save_path)

            text = " ".join(s.text for s in segments_list).strip()
            fluency = analyze_fluency(segments_list, info.duration)

            t_done = time.perf_counter()
            transcribe_time = t_done - t_saved
            total_time = t_done - t_start

            print(
                f"[DONE] {filename} | audio={info.duration:.1f}s | "
                f"transcribe={transcribe_time:.1f}s | total={total_time:.1f}s | "
                f"wpm={fluency['words_per_minute']}"
            )

            return {
                "text": text,
                "duration": round(info.duration, 2),
                "saved_as": filename,
                "size_kb": round(size_kb, 1),
                "fluency": fluency,
                "timings": {
                    "save_ms": round(save_time * 1000),
                    "transcribe_s": round(transcribe_time, 2),
                    "total_s": round(total_time, 2),
                },
            }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] {filename} | {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка транскрибации: {str(e)}")
    finally:
        # Файл НЕ удаляем сразу - оставляем для отладки.
        # В production можно удалять через фоновую задачу (см. ниже).
        pass