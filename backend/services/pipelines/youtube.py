"""YouTube transcript pipeline.

Fetches a video's transcript (auto-generated or uploaded captions) via the
free youtube-transcript-api library - no API key, no quota - and wraps it as
a single Document. The transcript then flows into the same
    chunk -> embed -> ChromaDB -> retrieve -> chain
path as a PDF page would.

Uses the youtube-transcript-api >= 1.0 API:
  api = YouTubeTranscriptApi(); fetched = api.fetch(video_id, languages=[...])
  fetched is iterable; each snippet has .text / .start / .duration as ATTRIBUTES
  (the pre-1.0 dict-access pattern no longer works).
"""
import re
from datetime import datetime, timezone

from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
)


# Matches the 11-character video ID in any standard YouTube URL shape:
#   youtube.com/watch?v=ID , youtu.be/ID , youtube.com/embed/ID , .../shorts/ID
_VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)"
    r"([A-Za-z0-9_-]{11})"
)


def _extract_video_id(url: str) -> str:
    match = _VIDEO_ID_RE.search(url)
    if not match:
        raise ValueError(f"Could not extract a YouTube video ID from: {url}")
    return match.group(1)


def load_youtube_transcript(
    url: str,
    languages: list[str] | None = None,
) -> list[Document]:
    """Fetch a YouTube transcript and return one Document.

    A single Document is the right granularity: the transcript is one
    continuous stream of dialogue, not naturally paginated. The
    RecursiveCharacterTextSplitter chunks it downstream like PDF text.
    """
    if languages is None:
        languages = ["en"]

    video_id = _extract_video_id(url)
    upload_time = datetime.now(timezone.utc).isoformat()

    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=languages)
    except TranscriptsDisabled:
        raise RuntimeError(f"Transcripts are disabled for video {video_id}.")
    except NoTranscriptFound:
        raise RuntimeError(
            f"No transcript found for video {video_id} in languages {languages}."
        )

    # FetchedTranscript is iterable; each snippet is a FetchedTranscriptSnippet
    # with .text as an attribute (not a dict key).
    text = " ".join(snippet.text for snippet in fetched)

    return [Document(
        page_content=text,
        metadata={
            "doc_name": f"youtube:{video_id}",
            "doc_type": "youtube",
            "video_id": video_id,
            "video_url": f"https://www.youtube.com/watch?v={video_id}",
            "content_type": "transcript",
            "upload_time": upload_time,
        },
    )]