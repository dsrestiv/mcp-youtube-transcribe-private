import logging
import os
import whisper
import yt_dlp
from yt_dlp.utils import DownloadError

# This will get the logger that was configured in mcp_server.py
logger = logging.getLogger(__name__)

# A simple cache for the Whisper model so we don't reload it every time
_whisper_model = None


def _yt_dlp_hook(d):
    """
    A hook for yt-dlp to capture its progress. We use this to prevent it from
    printing directly to stdout, which would corrupt the JSON-RPC communication.
    """
    if d['status'] == 'finished':
        # The 'filename' key is provided once the download is complete.
        logger.info(f"yt-dlp hook: Finished downloading '{d.get('filename', 'unknown file')}'")
    # We ignore the 'downloading' status to prevent log spam.


def get_youtube_transcript(query: str, force_whisper: bool = False) -> dict:
    """
    Searches for a YouTube video, downloads it, and returns the transcript.
    It will try to get an official transcript first, unless force_whisper is True.
    """
    global _whisper_model
    logger.info(f"get_youtube_transcript called with query: '{query}', force_whisper: {force_whisper}")

    try:
        # --- 1. Search for the video and get its info ---
        logger.info("Searching for video with yt-dlp...")
        ydl_opts_info = {
            'format': 'bestaudio/best',
            'noplaylist': True,
            'default_search': 'ytsearch',
            'quiet': True,
            'noprogress': True,
            'logger': logger,
        }
        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            info = ydl.extract_info(query, download=False)
            video_info = info['entries'][0] if 'entries' in info else info

        video_url = video_info.get("webpage_url")
        video_title = video_info.get("title", "Unknown Title")
        logger.info(f"Found video: '{video_title}' at {video_url}")

        transcript_text = None
        transcript_source = "Not Available"

        # --- 2. Try to get the official transcript ---
        if not force_whisper:
            logger.info("Official transcript check skipped for this debug version. Proceeding to Whisper.")

        # --- 3. Use Whisper if needed ---
        if transcript_text is None:
            transcript_source = "Whisper (AI Generated)"

            output_dir = os.path.join(os.path.dirname(__file__), "testing", "audio_cache")
            os.makedirs(output_dir, exist_ok=True)
            audio_path = os.path.join(output_dir, f"{video_info.get('id', 'default_id')}.mp3")

            logger.info(f"Downloading audio to: {audio_path}")
            ydl_opts_audio = {
                'format': 'bestaudio/best',
                'outtmpl': audio_path,
                'quiet': True,
                'noprogress': True,
                'overwrites': True,
                'logger': logger,
                # This hook will robustly silence stdout messages
                'progress_hooks': [_yt_dlp_hook],
            }
            with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
                ydl.download([video_url])
            logger.info("Audio download complete.")

            # Load the Whisper model if it's not already in memory
            if _whisper_model is None:
                logger.info("Whisper model not loaded. Loading 'tiny' model now...")
                _whisper_model = whisper.load_model("tiny")
                logger.info("Whisper model 'tiny' loaded successfully.")

            # Run the transcription
            logger.info("Starting Whisper transcription...")
            result = _whisper_model.transcribe(audio_path, fp16=False)
            transcript_text = result["text"]
            logger.info("Whisper transcription complete.")

        return {
            "status": "success",
            "title": video_title,
            "url": video_url,
            "source": transcript_source,
            "transcript": transcript_text
        }

    except DownloadError as e:
        logger.error(f"yt-dlp download error: {e}", exc_info=True)
        return {"status": "error", "message": f"Could not find or download the video: {e}"}
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_youtube_transcript: {e}", exc_info=True)
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}
