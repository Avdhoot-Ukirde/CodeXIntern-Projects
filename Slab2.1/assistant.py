"""
Voice-Activated Personal Assistant (Python)
------------------------------------------
Features implemented:
- Voice input (SpeechRecognition + default Google Web Speech API)
- Text-to-speech output (pyttsx3 -> offline on Windows/mac/Linux)
- Weather: city → geocode (OpenStreetMap / Nominatim), forecast via Open-Meteo (no API key)
- News: headlines via Google News RSS (no API key)
- Reminders: one-off reminders persisted in SQLite and scheduled with APScheduler
- Simple intent parser (natural-ish commands like: "weather in Pune", "read the news",
  "remind me in 10 minutes to drink water", "set a reminder at 5:30 pm to call mom",
  "list reminders", "delete reminder 3", etc.)
- Fallback to text-only mode if microphone not available

HOW TO RUN (quick):
1) Create & activate a virtual environment
   - Windows (PowerShell):
       python -m venv .venv
       .venv\\Scripts\\Activate.ps1
   - macOS/Linux:
       python3 -m venv .venv
       source .venv/bin/activate

2) Install dependencies
   pip install --upgrade pip
   pip install SpeechRecognition pyttsx3 pyaudio requests feedparser geopy APScheduler python-dateutil

   If PyAudio fails on Windows, try:
   pip install pipwin
   pipwin install pyaudio

3) Run
   python assistant.py

4) At startup, choose mode:
   - voice  : press Enter then speak when prompted
   - text   : type commands directly

Example commands to try:
- "what's the weather in Pune"
- "read the news"
- "technology headlines"
- "remind me in 1 minute to drink water"
- "set a reminder at 5:45 pm to stretch"
- "list reminders"
- "delete reminder 2"
- "time" / "date" / "help" / "quit"

NOTE: This is a baseline you can extend (alarms, timers, jokes, music control,
email/calendar integration, hotword detection, offline STT via Vosk, etc.).
"""

import os
import re
import sys
import time
import json
import math
import queue
import sqlite3
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass

# Third‑party libraries
import requests               # HTTP for weather
import feedparser             # RSS for news
from geopy.geocoders import Nominatim  # Geocoding city -> lat/lon
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from dateutil import tz
from dateutil import parser as dtparser

try:
    import speech_recognition as sr  # STT (cloud via Google by default)
except Exception as e:
    sr = None

try:
    import pyttsx3  # TTS (offline)
except Exception as e:
    pyttsx3 = None

# -------------------------
# Utility / configuration
# -------------------------
APP_NAME = "PyVoice Assistant"
DB_PATH = os.path.join(os.path.dirname(__file__), "reminders.db")
USER_TZ = tz.gettz()  # Auto-detect local timezone

# Open-Meteo: we will request current weather + next hours data (no API key)
OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

# Google News RSS (India English by default; adjust as needed)
GOOGLE_NEWS_RSS = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"
GOOGLE_NEWS_TECH_RSS = "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-IN&gl=IN&ceid=IN:en"

# Mapping Open-Meteo weather codes -> human-friendly summary
# Ref: https://open-meteo.com/en/docs
WEATHER_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

# -------------------------
# Text-to-Speech wrapper
# -------------------------
class TTS:
    def __init__(self, rate: int = 180, volume: float = 1.0):
        """Initialize pyttsx3 engine with a comfortable speaking rate.
        Works offline using system voices (SAPI5 on Windows, NSSpeech on macOS, eSpeak on Linux).
        """
        self.engine = None
        if pyttsx3:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', rate)
                self.engine.setProperty('volume', volume)
                # Optional: pick a female/male voice if available
                voices = self.engine.getProperty('voices')
                # Try to select an English voice
                for v in voices:
                    if 'en' in (v.languages[0].decode() if isinstance(v.languages[0], bytes) else str(v.languages[0]).lower()):
                        self.engine.setProperty('voice', v.id)
                        break
            except Exception as e:
                print(f"[TTS] Failed to init pyttsx3: {e}")
                self.engine = None
        else:
            print("[TTS] pyttsx3 not installed; voice output disabled")

    def say(self, text: str):
        """Speak text aloud; also print to console for visibility."""
        safe = text.strip()
        print(f"Assistant: {safe}")
        if self.engine:
            try:
                self.engine.say(safe)
                self.engine.runAndWait()
            except Exception as e:
                print(f"[TTS] Error speaking: {e}")

# -------------------------
# Speech-to-Text wrapper
# -------------------------
class STT:
    def __init__(self):
        """A simple SR (SpeechRecognition) based listener.
        Uses default Google Web Speech recognizer (free tier, rate-limited). Requires internet.
        """
        self.recognizer = None
        self.mic = None
        if sr:
            try:
                self.recognizer = sr.Recognizer()
                # Tweak for better results
                self.recognizer.energy_threshold = 300  # default 300; adapt if needed
                self.recognizer.dynamic_energy_threshold = True
                self.mic = sr.Microphone()
            except Exception as e:
                print(f"[STT] Microphone/Recognizer init failed: {e}")
                self.recognizer = None
                self.mic = None
        else:
            print("[STT] SpeechRecognition not installed; voice input disabled")

    def listen_once(self, prompt: str = "Speak now...") -> str | None:
        """Capture a single utterance and return transcribed text (lowercased).
        Returns None on failure.
        """
        if not (self.recognizer and self.mic):
            return None
        print(prompt)
        with self.mic as source:
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, phrase_time_limit=8)
                text = self.recognizer.recognize_google(audio)
                return text.strip()
            except sr.WaitTimeoutError:
                print("[STT] Timeout waiting for speech")
            except sr.UnknownValueError:
                print("[STT] Sorry, I didn't catch that.")
            except sr.RequestError as e:
                print(f"[STT] API error: {e}")
            except Exception as e:
                print(f"[STT] Error: {e}")
        return None

# -------------------------
# Reminders (SQLite + APScheduler)
# -------------------------
@dataclass
class Reminder:
    id: int
    text: str
    when_utc: datetime
    created_utc: datetime

class ReminderManager:
    def __init__(self, tts: TTS):
        self.tts = tts
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                when_utc TEXT NOT NULL,
                created_utc TEXT NOT NULL
            )
            """
        )
        self.conn.commit()
        self.scheduler = BackgroundScheduler(timezone="UTC")
        self.scheduler.start()
        self._reschedule_existing()

    def _reschedule_existing(self):
        """On startup, load all reminders in the future and schedule them."""
        cur = self.conn.execute("SELECT id, text, when_utc, created_utc FROM reminders")
        now_utc = datetime.utcnow().replace(tzinfo=tz.UTC)
        for rid, text, when_iso, created_iso in cur.fetchall():
            when_utc = datetime.fromisoformat(when_iso).replace(tzinfo=tz.UTC)
            if when_utc > now_utc:
                self._schedule_job(Reminder(rid, text, when_utc, datetime.fromisoformat(created_iso).replace(tzinfo=tz.UTC)))

    def _alert(self, reminder_id: int, text: str):
        self.tts.say(f"Reminder: {text}")
        # Optionally, delete the reminder after firing, or keep history. We'll keep it but you can clean up later.

    def _schedule_job(self, r: Reminder):
        trigger = DateTrigger(run_date=r.when_utc)
        self.scheduler.add_job(self._alert, trigger=trigger, args=[r.id, r.text], id=f"reminder_{r.id}", replace_existing=True)

    def add_reminder(self, text: str, when_local: datetime) -> Reminder:
        """Insert reminder and schedule it. `when_local` is in user's local tz; store UTC."""
        if when_local.tzinfo is None:
            when_local = when_local.replace(tzinfo=USER_TZ)
        when_utc = when_local.astimezone(tz.UTC)
        created_utc = datetime.utcnow().replace(tzinfo=tz.UTC)
        cur = self.conn.execute(
            "INSERT INTO reminders (text, when_utc, created_utc) VALUES (?, ?, ?)",
            (text, when_utc.isoformat(), created_utc.isoformat()),
        )
        self.conn.commit()
        rid = cur.lastrowid
        r = Reminder(rid, text, when_utc, created_utc)
        self._schedule_job(r)
        return r

    def list_reminders(self) -> list[Reminder]:
        cur = self.conn.execute("SELECT id, text, when_utc, created_utc FROM reminders ORDER BY when_utc ASC")
        res = []
        for rid, text, when_iso, created_iso in cur.fetchall():
            res.append(Reminder(rid, text, datetime.fromisoformat(when_iso).replace(tzinfo=tz.UTC), datetime.fromisoformat(created_iso).replace(tzinfo=tz.UTC)))
        return res

    def delete_reminder(self, rid: int) -> bool:
        cur = self.conn.execute("DELETE FROM reminders WHERE id = ?", (rid,))
        self.conn.commit()
        try:
            self.scheduler.remove_job(f"reminder_{rid}")
        except Exception:
            pass
        return cur.rowcount > 0

# -------------------------
# Weather client (geocode + Open-Meteo)
# -------------------------
class WeatherClient:
    def __init__(self):
        # Nominatim requires a user_agent string
        self.geocoder = Nominatim(user_agent=APP_NAME)

    def geocode_city(self, city: str) -> tuple[float, float, str] | None:
        """Return (lat, lon, display_name) for a city string, or None if not found."""
        try:
            loc = self.geocoder.geocode(city)
            if not loc:
                return None
            return float(loc.latitude), float(loc.longitude), loc.address
        except Exception as e:
            print(f"[Weather] Geocoding failed: {e}")
            return None

    def get_weather(self, city: str) -> str:
        """Return a human-friendly weather summary for the given city."""
        g = self.geocode_city(city)
        if not g:
            return f"I couldn't find '{city}'. Try a more specific name."
        lat, lon, place = g
        # Build Open-Meteo request
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "weather_code",
                "wind_speed_10m",
            ]),
            "timezone": "auto",
        }
        try:
            r = requests.get(OPEN_METEO_URL, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            cur = data.get("current", {})
            temp = cur.get("temperature_2m")
            feels = cur.get("apparent_temperature")
            wind = cur.get("wind_speed_10m")
            hum = cur.get("relative_humidity_2m")
            code = cur.get("weather_code")
            desc = WEATHER_CODE_MAP.get(int(code), "Unknown conditions") if code is not None else "Unknown conditions"
            city_name = place.split(",")[0] if place else city
            return (
                f"Weather in {city_name}: {desc}. Temperature {temp}°C (feels like {feels}°C), "
                f"humidity {hum}%, wind {wind} km/h."
            )
        except Exception as e:
            print(f"[Weather] Request failed: {e}")
            return "Sorry, I couldn't fetch the weather right now."

# -------------------------
# News client (RSS → headlines)
# -------------------------
class NewsClient:
    def __init__(self):
        self.default_feed = GOOGLE_NEWS_RSS
        self.tech_feed = GOOGLE_NEWS_TECH_RSS

    def headlines(self, topic: str | None = None, limit: int = 5) -> list[str]:
        """Return a list of headline strings (title + source)."""
        url = self.default_feed
        if topic and topic.lower() in {"tech", "technology", "it"}:
            url = self.tech_feed
        try:
            feed = feedparser.parse(url)
            items = []
            for entry in feed.entries[:limit]:
                title = getattr(entry, 'title', '').strip()
                source = getattr(entry, 'source', {}).get('title') if hasattr(entry, 'source') else None
                if not source:
                    # Try to parse source from title if present " - Source"
                    if " - " in title:
                        title, source = title.rsplit(" - ", 1)
                line = f"{title}" + (f" — {source}" if source else "")
                items.append(line)
            return items
        except Exception as e:
            print(f"[News] Failed to fetch headlines: {e}")
            return []

# -------------------------
# Intent parsing helpers
# -------------------------
REMIND_IN_RE = re.compile(r"remind me(?: to)? (?P<task>.+?) in (?P<num>\d+)\s*(?P<unit>minutes?|hours?)", re.IGNORECASE)
REMIND_AT_RE = re.compile(r"(set a )?reminder (for |at )(?P<time>[^ ]+(?:\s*[ap]m)?) (to )?(?P<task>.+)", re.IGNORECASE)
REMIND_SIMPLE_RE = re.compile(r"remind me (?:to )?(?P<task>.+) at (?P<time>.+)", re.IGNORECASE)
DELETE_REM_RE = re.compile(r"(delete|remove|cancel) reminder (?P<id>\d+)", re.IGNORECASE)
WEATHER_RE = re.compile(r"(weather|temperature) (in|at|for) (?P<city>.+)", re.IGNORECASE)
NEWS_RE = re.compile(r"(news|headlines)(?: about| on| for)? (?P<topic>\w+)?", re.IGNORECASE)
TIME_RE = re.compile(r"^(time|what's the time|current time)$", re.IGNORECASE)
DATE_RE = re.compile(r"^(date|what's the date|today's date)$", re.IGNORECASE)
LIST_REMS_RE = re.compile(r"(list|show) reminders", re.IGNORECASE)
HELP_RE = re.compile(r"^help$", re.IGNORECASE)
QUIT_RE = re.compile(r"^(quit|exit|bye)$", re.IGNORECASE)


def parse_time_local(text_time: str) -> datetime | None:
    """Parse a time expression like "5:30 pm" / "17:30" / "tomorrow 8am" into a localized datetime.
    Uses python-dateutil's parser with today's date as reference.
    If the parsed time is in the past for today, schedule for tomorrow.
    """
    try:
        now = datetime.now(tz=USER_TZ)
        dt = dtparser.parse(text_time, default=now)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=USER_TZ)
        # If the time is before now, push to next day
        if dt <= now:
            dt = dt + timedelta(days=1)
        return dt
    except Exception as e:
        print(f"[ParseTime] Failed to parse '{text_time}': {e}")
        return None

# -------------------------
# Core Assistant logic
# -------------------------
class Assistant:
    def __init__(self):
        self.tts = TTS()
        self.stt = STT()
        self.weather = WeatherClient()
        self.news = NewsClient()
        self.rems = ReminderManager(self.tts)

    def help_text(self) -> str:
        return (
            "You can say things like:\n"
            "- what's the weather in <city>\n"
            "- read the news / technology news\n"
            "- remind me in 10 minutes to <task>\n"
            "- set a reminder at 5:30 pm to <task>\n"
            "- list reminders / delete reminder <id>\n"
            "- time / date / help / quit"
        )

    def handle(self, utterance: str) -> bool:
        """Handle an utterance. Returns False to request exit, True to continue."""
        if not utterance:
            self.tts.say("I didn't hear anything. Please try again.")
            return True
        text = utterance.strip()
        low = text.lower()

        # Quit
        if QUIT_RE.match(low):
            self.tts.say("Goodbye!")
            return False

        # Help
        if HELP_RE.match(low):
            self.tts.say(self.help_text())
            return True

        # Time / Date
        if TIME_RE.match(low):
            now = datetime.now(tz=USER_TZ)
            self.tts.say(f"It's {now.strftime('%I:%M %p')}")
            return True
        if DATE_RE.match(low):
            today = datetime.now(tz=USER_TZ)
            self.tts.say(f"Today is {today.strftime('%A, %B %d, %Y')}")
            return True

        # Weather
        m = WEATHER_RE.search(text)
        if m:
            city = m.group('city')
            resp = self.weather.get_weather(city)
            self.tts.say(resp)
            return True

        # News
        m = NEWS_RE.search(text)
        if m:
            topic = m.group('topic')
            items = self.news.headlines(topic=topic, limit=5)
            if not items:
                self.tts.say("Sorry, I couldn't fetch headlines right now.")
            else:
                self.tts.say("Here are the top headlines:")
                for i, line in enumerate(items, 1):
                    self.tts.say(f"{i}. {line}")
            return True

        # Reminders: "remind me in X minutes/hours to <task>"
        m = REMIND_IN_RE.search(text)
        if m:
            task = m.group('task').strip()
            num = int(m.group('num'))
            unit = m.group('unit').lower()
            delta = timedelta(minutes=num) if unit.startswith('min') else timedelta(hours=num)
            when = datetime.now(tz=USER_TZ) + delta
            r = self.rems.add_reminder(task, when)
            local_time = when.strftime('%I:%M %p')
            self.tts.say(f"Okay, I'll remind you at {local_time} to {task}.")
            return True

        # Reminders: "set a reminder at 5:30 pm to call mom"
        m = REMIND_AT_RE.search(text) or REMIND_SIMPLE_RE.search(text)
        if m:
            task = m.group('task').strip()
            time_text = m.group('time').strip()
            when = parse_time_local(time_text)
            if when:
                r = self.rems.add_reminder(task, when)
                self.tts.say(f"Reminder set for {when.strftime('%I:%M %p')} to {task}.")
            else:
                self.tts.say("Sorry, I couldn't understand the time.")
            return True

        # List reminders
        if LIST_REMS_RE.search(text):
            items = self.rems.list_reminders()
            if not items:
                self.tts.say("You have no reminders.")
            else:
                self.tts.say("Here are your reminders:")
                for r in items:
                    local = r.when_utc.astimezone(USER_TZ)
                    self.tts.say(f"{r.id}: {r.text} at {local.strftime('%I:%M %p on %b %d')}.")
            return True

        # Delete reminder by ID
        m = DELETE_REM_RE.search(text)
        if m:
            rid = int(m.group('id'))
            ok = self.rems.delete_reminder(rid)
            if ok:
                self.tts.say(f"Deleted reminder {rid}.")
            else:
                self.tts.say(f"I couldn't find reminder {rid}.")
            return True

        # Unknown intent → hint
        self.tts.say("I can help with weather, news, and reminders. Say 'help' to see examples.")
        return True

    def loop_voice(self):
        if not (self.stt and self.stt.mic):
            self.tts.say("Microphone is not available. Please use text mode.")
            return
        self.tts.say("Voice mode: press Enter and speak after the beep. Say 'quit' to exit.")
        while True:
            try:
                input("\n[Press Enter to speak]")
            except (EOFError, KeyboardInterrupt):
                break
            # Small auditory cue (print only); you can add a real beep if desired
            print("* Beep *")
            said = self.stt.listen_once()
            if said:
                print(f"You said: {said}")
            cont = self.handle(said or "")
            if not cont:
                break

    def loop_text(self):
        self.tts.say("Text mode: type your command. Type 'help' for examples, 'quit' to exit.")
        while True:
            try:
                line = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            cont = self.handle(line)
            if not cont:
                break


def main():
    print(f"=== {APP_NAME} ===")
    print("Choose input mode: [1] Voice  [2] Text (default: 2)")
    choice = input("> ").strip()
    assistant = Assistant()
    if choice == "1":
        assistant.loop_voice()
    else:
        assistant.loop_text()


if __name__ == "__main__":
    # When this file is named assistant.py and run as described in the header,
    # you'll have a working baseline voice assistant.
    main()
