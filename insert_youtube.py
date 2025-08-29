import os
import argparse
import subprocess
import yt_dlp
import torch
import shutil
import time
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from faster_whisper import WhisperModel
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import (
    compute_hash,
    document_exists,
    get_chroma_client,
    get_or_create_collection,
    log_collection_contents,
    chunk_text
)

__ffsig__ = "S.MFFS-X9Δ7"

# === CONFIGURABLE SETTINGS ===
MAX_CONCURRENT_TASKS = 15
WHISPER_MODEL_SIZE = "small"
USE_GPU = torch.cuda.is_available()
# =============================

def get_video_urls(channel_url):
    ydl_opts = {
        'quiet': False,
        'extract_flat': True,
        'force_generic_extractor': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
        return [entry['url'] for entry in info.get('entries', [])]

def flatten_to_strings(x: any) -> list[str]:
    if isinstance(x, str):
        return [x]
    elif isinstance(x, (list, tuple)):
        result = []
        for item in x:
            result.extend(flatten_to_strings(item))
        return result
    else:
        return [str(x)]

def get_transcript_youtube_api(video_id):
    try:
        time.sleep(2)  # add delay to avoid 429s
        return " ".join([
            entry['text']
            for entry in YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        ])
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        tqdm.write(f" YouTube API failed for {video_id}: {e}")
        return None
    except Exception as e:
        print(f"Unhandled error for video {video_id}: {e}")
        return ""

def download_and_transcribe(video_url, video_id):
    import yt_dlp
    from faster_whisper import WhisperModel

    def download_audio():
        output_template = f"{video_id}.%(ext)s"
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'quiet': True,
                'outtmpl': output_template,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            return f"{video_id}.mp3"
        except Exception:
            return None

    audio_path = download_audio()
    if not audio_path or not os.path.exists(audio_path):
        return None

    try:
        model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cuda" if USE_GPU else "cpu",
            compute_type="float16" if USE_GPU else "int8"
        )
        segments, _ = model.transcribe(audio_path, beam_size=5)
        os.remove(audio_path)
        return " ".join([segment.text for segment in segments])
    except Exception:
        return None

def process_video_sync(args):
    video_url, collection_name, db_dir = args
    import time
    from tqdm import tqdm

    video_id = video_url.split("v=")[-1]
    start_time = time.time()
    tqdm.write(f"[{video_id}] Starting processing")

    text = get_transcript_youtube_api(video_id)
    if text:
        tqdm.write(f"[{video_id}] Transcript fetched from YouTube API")
    else:
        tqdm.write(f"[{video_id}] Transcript not available on YouTube. Using Whisper...")
        text = download_and_transcribe(video_url, video_id)
        if text:
            tqdm.write(f"[{video_id}] Transcription completed with Whisper")
        else:
            tqdm.write(f"[{video_id}] Failed to retrieve transcript")
            return []

    collection = get_or_create_collection(get_chroma_client(db_dir), collection_name)
    chunks = chunk_text(text)
    # Flatten and clean any bad chunks
    flattened_chunks = []
    for chunk in chunks:
        if isinstance(chunk, (list, tuple)):
            cleaned = " ".join(str(c).strip() for c in chunk if isinstance(c, str))
        elif isinstance(chunk, str):
            cleaned = chunk.strip()
        else:
            cleaned = str(chunk).strip()

        if cleaned:
            flattened_chunks.append(cleaned)

    chunks = flattened_chunks
    tqdm.write(f"[{video_id}] Text chunked into {len(chunks)} segments")

    new_chunks = []
    for i, chunk in enumerate(chunks):
        doc_hash = compute_hash(chunk)
        if document_exists(collection, doc_hash):
            tqdm.write(f"[{video_id}] Skipping duplicate chunk {i} ({doc_hash})")
            continue
        metadata = {"source": video_url, "doc_hash": doc_hash, "chunk_index": i}
        # --- Robust: Ensure chunk is a single clean string ---
        if isinstance(chunk, tuple):
            chunk = " ".join(str(part).strip() for part in chunk if isinstance(part, str))
        elif not isinstance(chunk, str):
            chunk = str(chunk).strip()

        # Only include non-empty, valid chunks
        if chunk:
            new_chunks.append((doc_hash, chunk, metadata))

    tqdm.write(f"[{video_id}] {len(new_chunks)} new chunks ready for insertion")
    tqdm.write(f"[{video_id}] Finished in {time.time() - start_time:.2f} seconds")
    return new_chunks

def process_youtube_channel(channel_url, collection_name, db_dir):
    video_urls = get_video_urls(channel_url)
    tqdm.write(f" Found {len(video_urls)} videos.")
    all_chunks = []

    args = [(url, collection_name, db_dir) for url in video_urls]

    with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_TASKS) as executor:
        futures = [executor.submit(process_video_sync, arg) for arg in args]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            try:
                result = f.result()
                all_chunks.extend(result)
            except Exception as e:
                tqdm.write(f"[error] Task failed: {e}")

    if not all_chunks:
        tqdm.write("No new content to insert.")
        return

    client = get_chroma_client(db_dir)
    collection = get_or_create_collection(client, collection_name)

    # ── Monkey‑patch add() to force every document into a flat string ──
    orig_add = collection.add
    def add_strict(documents, metadatas, ids, *args, **kwargs):
        # Coerce every document into one string
        clean_docs = []
        for d in documents:
            if isinstance(d, (list, tuple)):
                # join any nested lists/tuples of strings
                clean_docs.append(" ".join(str(x) for x in d))
            else:
                clean_docs.append(str(d))
        # Now call the real add()
        return orig_add(documents=clean_docs, metadatas=metadatas, ids=ids, *args, **kwargs)

    collection.add = add_strict

    # coerce every document to a string—no chance of a tuple sneaking through
    documents = [str(c[1]) for c in all_chunks]
    metadatas = [c[2] for c in all_chunks]
    ids = [c[0] for c in all_chunks]

    # Get existing IDs from the collection
    existing_items = collection.peek(limit=50000)  # Adjust limit as needed
    existing_ids = set(existing_items["ids"]) if existing_items and "ids" in existing_items else set()

    # Filter out already existing IDs, keep 1:1 mapping
    filtered = [
        (i, d, m)
        for i, d, m in zip(ids, documents, metadatas)
        if i not in existing_ids
    ]

    if not filtered:
        tqdm.write("No new unique chunks to insert (all IDs already exist).")
        return

    # Insert one at a time (guarantees len(ids)==len(docs))
    inserted = 0
    for i, d, m in filtered:
        # force string
        if not isinstance(d, str):
            d = str(d)
        try:
            collection.add(documents=[d], metadatas=[m], ids=[i])
            inserted += 1
        except Exception as e:
            tqdm.write(f"[error] Failed to insert chunk {i}: {e}")

    tqdm.write(f"Successfully embedded {inserted} new unique chunks from YouTube.")

def main():
    parser = argparse.ArgumentParser(description="Insert YouTube transcripts into ChromaDB")
    parser.add_argument("channel_url", help="YouTube channel or playlist URL")
    parser.add_argument("--collection", default="youtube", help="ChromaDB collection name")
    parser.add_argument("--db-dir", default="./chroma_db", help="ChromaDB directory")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--clear-db", action="store_true", help="Clear ChromaDB directory before insert")
    parser.add_argument("--log", action="store_true", help="Log stored content after insert")
    args = parser.parse_args()

    if args.clear_db and os.path.exists(args.db_dir):
        tqdm.write(f"[info] Clearing ChromaDB directory: {args.db_dir}")
        shutil.rmtree(args.db_dir)

    process_youtube_channel(args.channel_url, args.collection, args.db_dir)

    if args.log:
        tqdm.write(f"\n[log] Previewing content from ChromaDB collection '{args.collection}':")
        log_collection_contents(get_or_create_collection(get_chroma_client(args.db_dir), args.collection))

if __name__ == "__main__":
    main()