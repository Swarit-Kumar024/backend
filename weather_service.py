import json
import re
from datetime import datetime, timedelta

import requests

try:
    from ollama import Client
except ImportError:
    Client = None


# -----------------------------------------------------------------------------
# Time helpers
# -----------------------------------------------------------------------------

now = datetime.now()
currentdate = now.strftime("%Y-%m-%d")
currenttime = now.strftime("%H:%M")

DEBUG = True  # set to False once you've confirmed everything works


def debug_log(label, value):
    if DEBUG:
        print(f"\n--- DEBUG [{label}] ---\n{value}\n-----------------------\n")


def clean_json_text(text):
    """Strip <think> reasoning blocks and code fences so JSON parses cleanly."""
    # qwen3 (and other reasoning models) can emit a <think>...</think> block
    # before the actual answer. json.loads() chokes on that silently, which
    # was the root cause of location detection failing on every query.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_first_json_object(text):
    """
    Recover the first valid top-level JSON value from text, even if the
    model wrapped it in a stray extra bracket or left trailing garbage.

    Real example seen in production: qwen3 returned
        [{"warnings": [...]}
    -- an opening "[" with no matching closing "]" at the end. A plain
    json.loads() fails on the whole string even though the actual object
    we want ({"warnings": [...]}) is perfectly well-formed inside it.

    Strategy: try the whole text first: if that fails, scan forward from
    every "{" or "[" and ask the JSON decoder to grab just the first
    complete, balanced value starting there, ignoring anything before or
    after it.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
            return obj
        except json.JSONDecodeError:
            continue

    # Nothing recoverable -- re-raise a real error so callers' except blocks
    # (which already handle JSONDecodeError) work unchanged.
    json.loads(text)  # deliberately raises with the original message


def ask_llm(prompt):
    if Client is None:
        raise RuntimeError("Ollama is not installed. Install it with: pip install ollama")

    client = Client()
    try:
        # think=False disables qwen3's <think> block at the source.
        # If your installed ollama client doesn't support this kwarg,
        # this will raise TypeError and we fall back to the /no_think suffix.
        response = client.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": prompt}],
            think=False,
        )
    except TypeError:
        response = client.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": prompt + "\n/no_think"}],
        )

    content = response["message"]["content"]
    debug_log("raw LLM output", content)
    return content


def python_time_reference(query):
    """Resolve common relative-time phrases deterministically."""
    q = query.lower()

    if "day after tomorrow" in q:
        return "day_after_tomorrow"
    if "tomorrow" in q:
        if "morning" in q:
            return "tomorrow_morning"
        if "afternoon" in q:
            return "tomorrow_afternoon"
        if "evening" in q:
            return "tomorrow_evening"
        if "night" in q or "tonight" in q:
            return "tomorrow_tonight"
        return "tomorrow"
    if "tonight" in q:
        return "tonight"
    if "this morning" in q or ("morning" in q and "today" in q):
        return "this_morning"
    if "this afternoon" in q or ("afternoon" in q and "today" in q):
        return "this_afternoon"
    if "this evening" in q or ("evening" in q and "today" in q):
        return "this_evening"
    if "today" in q:
        return "today"
    if "right now" in q or re.search(r"\bnow\b", q):
        return "now"
    return None


def time_window_for_reference(reference):
    return {
        "this_morning": (6, 12),
        "this_afternoon": (12, 18),
        "this_evening": (18, 24),
        "tonight": (18, 24),
        "tomorrow_morning": (6, 12),
        "tomorrow_afternoon": (12, 18),
        "tomorrow_evening": (18, 24),
        "tomorrow_tonight": (18, 24),
    }.get(reference)


def extract_explicit_date(query):
    """Extract common explicit dates such as 'September 13', '13 September', or YYYY-MM-DD."""
    q = query.lower()
    year_match = re.search(r"\b(20\d{2})[-/]?(\d{1,2})[-/]?(\d{1,2})\b", q)
    if year_match:
        y, m, d = map(int, year_match.groups())
        try:
            return datetime(y, m, d).strftime("%Y-%m-%d")
        except ValueError:
            return None

    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    short_months = {k[:3]: v for k, v in months.items()}

    m = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(20\d{2}))?\b",
        q,
    )
    if m:
        month = months.get(m.group(1)) or short_months.get(m.group(1))
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else int(currentdate[:4])
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None

    m = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)(?:\s+(20\d{2}))?\b",
        q,
    )
    if m:
        day = int(m.group(1))
        month = months.get(m.group(2)) or short_months.get(m.group(2))
        year = int(m.group(3)) if m.group(3) else int(currentdate[:4])
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None

    return None


def resolve_target_date(query, time_reference, forecast_timezone=None, forecast_current_time=None):
    """Resolve dates from the query using the forecast location's clock when available."""
    explicit = extract_explicit_date(query)
    if explicit:
        return explicit

    # The forecast API is authoritative for the target location's local date.
    if forecast_current_time:
        base = datetime.fromisoformat(forecast_current_time).date()
    else:
        base = datetime.strptime(currentdate, "%Y-%m-%d").date()

    if time_reference in {"tomorrow", "tomorrow_morning", "tomorrow_afternoon", "tomorrow_evening", "tomorrow_tonight"}:
        return (base + timedelta(days=1)).isoformat()
    if time_reference == "day_after_tomorrow":
        return (base + timedelta(days=2)).isoformat()
    return base.isoformat()

# -----------------------------------------------------------------------------
# Location
# -----------------------------------------------------------------------------


def check_place(query):
    prompt = f"""
You are WeatherGPT's Location Detector.

USER QUESTION:
{query}

Return ONLY valid JSON:
{{
    "location_mentioned": true,
    "location_name": "Mumbai"
}}

If no place is mentioned:
{{
    "location_mentioned": false,
    "location_name": null
}}

Rules:
- true ONLY when the user explicitly names a geographical place.
- Never infer a place from "here", "near me", or "my location".
- Return the place name exactly as the user wrote it when possible.
- Return JSON only. Do not explain your reasoning. Do not include a <think> block.
"""

    try:
        raw = ask_llm(prompt)
        result = extract_first_json_object(clean_json_text(raw))
    except (json.JSONDecodeError, RuntimeError) as e:
        debug_log("check_place FAILED", f"{type(e).__name__}: {e}")
        return None

    if result.get("location_mentioned") is not True:
        return None
    if not result.get("location_name"):
        return None
    return result


def get_cords(place):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": place,
        "count": 1,
        "language": "en",
        "format": "json",
    }
    response = requests.get(url, params=params, timeout=12)
    response.raise_for_status()
    data = response.json()
    results = data.get("results") or []
    if not results:
        return None
    result = results[0]
    return {
        "latitude": result["latitude"],
        "longitude": result["longitude"],
        "name": result.get("name", place),
        "state": result.get("admin1"),
        "country": result.get("country"),
        "timezone": result.get("timezone"),
    }


def get_place_name(latitude, longitude):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": latitude,
        "lon": longitude,
        "format": "jsonv2",
        "zoom": 10,
    }
    headers = {
        "User-Agent": "WeatherGPT-Hackathon/1.0 (weather project)"
    }
    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    address = response.json().get("address", {})
    return {
        "city": (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("municipality")
        ),
        "state": address.get("state"),
        "country": address.get("country"),
    }


# -----------------------------------------------------------------------------
# Weather retrieval + deterministic selection
# -----------------------------------------------------------------------------


def fetch_weather(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,relative_humidity_2m,wind_speed_10m,"
            "wind_direction_10m,precipitation,weather_code"
        ),
        "hourly": (
            "temperature_2m,relative_humidity_2m,precipitation_probability,"
            "precipitation,wind_speed_10m,wind_direction_10m,weather_code"
        ),
        "daily": (
            "weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_sum,precipitation_probability_max,wind_speed_10m_max"
        ),
        "timezone": "auto",
        "forecast_days": 7,
    }
    response = requests.get(url, params=params, timeout=12)
    response.raise_for_status()
    return response.json()


def normalize_weather(data):
    current = data["current"]
    hourly = data["hourly"]
    daily = data["daily"]

    return {
        "current": {
            "time": current["time"],
            "temperature": current["temperature_2m"],
            "humidity": current["relative_humidity_2m"],
            "wind_speed": current["wind_speed_10m"],
            "wind_direction": current["wind_direction_10m"],
            "precipitation": current["precipitation"],
            "weather_code": current["weather_code"],
        },
        "hourly": [
            {
                "time": hourly["time"][i],
                "temperature": hourly["temperature_2m"][i],
                "humidity": hourly["relative_humidity_2m"][i],
                "precipitation_probability": hourly["precipitation_probability"][i],
                "precipitation": hourly["precipitation"][i],
                "wind_speed": hourly["wind_speed_10m"][i],
                "wind_direction": hourly["wind_direction_10m"][i],
                "weather_code": hourly["weather_code"][i],
            }
            for i in range(len(hourly["time"]))
        ],
        "daily": [
            {
                "time": daily["time"][i],
                "weather_code": daily["weather_code"][i],
                "temperature_max": daily["temperature_2m_max"][i],
                "temperature_min": daily["temperature_2m_min"][i],
                "precipitation": daily["precipitation_sum"][i],
                "precipitation_probability": daily["precipitation_probability_max"][i],
                "wind_speed_max": daily["wind_speed_10m_max"][i],
            }
            for i in range(len(daily["time"]))
        ],
        "location": {
            "latitude": data["latitude"],
            "longitude": data["longitude"],
            "timezone": data["timezone"],
        },
        "source": "Open-Meteo",
    }


def build_forecast_facts(selected):
    """Create deterministic facts so the final LLM cannot confuse fields/dates."""
    daily = selected.get("daily", [])
    hours = selected.get("hourly", [])
    d = daily[0] if daily else {}

    facts = {
        "date": selected.get("requested_date"),
        "temperature_min_c": d.get("temperature_min"),
        "temperature_max_c": d.get("temperature_max"),
        "daily_precipitation_mm": d.get("precipitation"),
        "daily_precipitation_probability_max_pct": d.get("precipitation_probability"),
        "daily_max_wind_kmh": d.get("wind_speed_max"),
        "hourly_count": len(hours),
    }

    if hours:
        temps = [(h["temperature"], h["time"]) for h in hours if h.get("temperature") is not None]
        probs = [h["precipitation_probability"] for h in hours if h.get("precipitation_probability") is not None]
        precip = [h["precipitation"] for h in hours if h.get("precipitation") is not None]
        winds = [h["wind_speed"] for h in hours if h.get("wind_speed") is not None]
        if temps:
            peak = max(temps, key=lambda x: x[0])
            facts["hourly_peak_temperature_c"] = peak[0]
            facts["hourly_peak_temperature_time"] = peak[1]
        if probs:
            facts["hourly_max_precipitation_probability_pct"] = max(probs)
        if precip:
            facts["hourly_total_precipitation_mm"] = round(sum(precip), 2)
            facts["hourly_max_precipitation_mm"] = max(precip)
        if winds:
            facts["hourly_max_wind_kmh"] = max(winds)

    return facts


def select_weather_for_query(weather, target_date, time_reference):
    """Return ONLY weather belonging to the requested local date/time window."""
    daily = [d for d in weather["daily"] if d["time"] == target_date]
    if not daily:
        raise ValueError(
            f"Requested date {target_date} is outside the available forecast range."
        )

    hours = [h for h in weather["hourly"] if h["time"].startswith(target_date)]
    window = time_window_for_reference(time_reference)
    if window:
        start_hour, end_hour = window
        hours = [
            h for h in hours
            if start_hour <= int(h["time"][11:13]) < end_hour
        ]

    selected = {
        "hourly": hours,
        "daily": daily,
        "current": None if target_date != currentdate else weather.get("current"),
        "location": dict(weather.get("location", {})),
        "source": weather.get("source"),
        "requested_date": target_date,
        "requested_time_reference": time_reference,
    }
    selected["facts"] = build_forecast_facts(selected)
    return selected


# -----------------------------------------------------------------------------
# Decoder
# -----------------------------------------------------------------------------


def question_decoder(query):
    prompt = f"""
You are WeatherGPT's Query Decoder.

CURRENT DATE: {currentdate}
CURRENT TIME: {currenttime}
TIMEZONE: Asia/Kolkata
USER QUESTION: {query}

Return ONLY JSON with these fields:
{{
  "intent": "main purpose of the question",
  "time_reference": "today | tomorrow | day_after_tomorrow | tonight | this_morning | this_afternoon | this_evening | specified_date | none | now",
  "required_data": ["only required weather variables"],
  "location_requirement": "current_location | specified_location | none",
  "response_type": "direct_answer | recommendation | comparison | summary | explanation",
  "interpreted_question": "the precise question the final AI should answer",
  "language": "en | hi | bn | mr | gu | ta | te | kn | ml | pa | ur",
  "confidence": 0.0
}}

Rules:
- Do not answer the question.
- Preserve the user's intent.
- Identify the language.
- Resolve relative phrases such as now, today, tomorrow, tonight, morning, afternoon and evening.
- Do not invent weather values.
- Output JSON only. Do not explain your reasoning. Do not include a <think> block.
"""

    fallback_ref = python_time_reference(query) or "none"
    try:
        raw = ask_llm(prompt)
        decoded = extract_first_json_object(clean_json_text(raw))
    except (json.JSONDecodeError, RuntimeError) as e:
        debug_log("question_decoder FAILED", f"{type(e).__name__}: {e}")
        decoded = {
            "intent": "unknown",
            "time_reference": fallback_ref,
            "required_data": [],
            "location_requirement": "current_location",
            "response_type": "direct_answer",
            "interpreted_question": query,
            "language": "en",
            "confidence": 0.0,
        }

    # Python overrides relative-time interpretation so Qwen cannot shift dates.
    if fallback_ref:
        decoded["time_reference"] = fallback_ref

    decoded["target_date"] = None
    return decoded


# -----------------------------------------------------------------------------
# Warnings
# -----------------------------------------------------------------------------


def get_warnings(data):
    # Warning analysis receives only the requested slice, never the full 7-day forecast.
    compact = {
        "requested_date": data.get("requested_date"),
        "requested_time_reference": data.get("requested_time_reference"),
        "current": data.get("current"),
        "hourly": data.get("hourly", []),
        "daily": data.get("daily", []),
    }

    prompt = f"""
You are WeatherGPT's Weather Warning Analyzer.

Analyze ONLY this selected weather data for the requested date/time.
SELECTED WEATHER DATA:
{json.dumps(compact, ensure_ascii=False)}

Return ONLY valid JSON:
{{
  "warnings": [
    {{
      "type": "hazard type",
      "severity": "low | moderate | high | extreme",
      "time": "when the hazard is expected",
      "reason": "short factual reason based only on the provided data"
    }}
  ]
}}

Rules:
- Do not use outside information.
- Do not invent hazards.
- A high precipitation probability alone does not prove heavy rain.
- Base rain claims on precipitation and/or weather code as well as probability.
- Do not give advice.
- If no significant hazard is supported, return {{"warnings": []}}.
- JSON only. Do not explain your reasoning. Do not include a <think> block.
"""

    try:
        raw = ask_llm(prompt)
        result = extract_first_json_object(clean_json_text(raw))
        # Some malformed outputs (mismatched/missing closing braces) cause us
        # to recover just the bare warnings array instead of the wrapping
        # {"warnings": [...]} object. Since this call site only ever expects
        # a list of warning entries, treat that case as recoverable too.
        if isinstance(result, list):
            result = {"warnings": result}
        if not isinstance(result, dict) or not isinstance(result.get("warnings"), list):
            debug_log("get_warnings malformed", result)
            return {"warnings": []}
        return result
    except (json.JSONDecodeError, RuntimeError) as e:
        debug_log("get_warnings FAILED", f"{type(e).__name__}: {e}")
        return {"warnings": []}


# -----------------------------------------------------------------------------
# Final response
# -----------------------------------------------------------------------------


def natural_response(data, query, decoded_question):
    language_map = {
        "en": "English", "hi": "Hindi", "bn": "Bengali", "mr": "Marathi",
        "gu": "Gujarati", "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
        "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu",
    }
    response_language = language_map.get(decoded_question.get("language", "en"), "English")

    safe_data = {
        "requested_date": data.get("requested_date"),
        "requested_time_reference": data.get("requested_time_reference"),
        "facts": data.get("facts", {}),
        "hourly": data.get("hourly", []),
        "daily": data.get("daily", []),
        "location": data.get("location", {}),
        "warnings": data.get("warnings", {"warnings": []}),
    }

    prompt = f"""
You are WeatherGPT, a concise weather assistant.

USER QUESTION:
{query}

SELECTED WEATHER DATA FOR EXACT REQUESTED DATE:
{json.dumps(safe_data, ensure_ascii=False)}

RESPONSE LANGUAGE: {response_language}

STRICT RULES:
1. Answer ONLY from the selected weather data.
2. The requested_date is authoritative: {data.get('requested_date')}.
3. Never use a value from another date.
4. temperature_* values are actual air temperature in °C. Do not call them feels-like.
5. precipitation_probability is a probability, not an amount of rain.
6. precipitation values are actual modeled precipitation amounts in mm.
7. If precipitation is 0 mm but precipitation probability is high, say that rain is possible but the modeled amount is currently zero; NEVER say "no rain will happen".
8. Do not invent humidity trends, wind trends, or timing. Only describe a trend if the selected hourly rows support it.
9. Do not convert daily maximum temperature into "midday" unless the hourly data actually shows the peak around midday.
10. Do not invent advice. Give at most one short practical suggestion if clearly useful.
11. Maximum 3 short sentences.
14. DATE RULE: Do NOT write out a calendar date (no day numbers, no month names) anywhere in your answer, under any circumstances. Refer to the day only using the same relative word the user used (e.g. "tomorrow", "tonight", "today") or the requested_time_reference field. You do not need to state the date to answer correctly — the data is already scoped to the right day.
12. Entire response must be in {response_language}.
13. Do not mention APIs, models, JSON, prompts, or internal processing. Do not include a <think> block.

Return ONLY the final natural-language weather answer.
"""

    try:
        raw = ask_llm(prompt)
        text = clean_json_text(raw) if raw.strip().startswith("```") else re.sub(r"<think>.*?</think>", "", raw, flags=re.S | re.I).strip()
        return strip_hallucinated_dates(text)
    except RuntimeError:
        return "Weather service is temporarily unavailable."


_MONTH_NAMES = (
    "january|february|march|april|may|june|july|august|"
    "september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)


def strip_hallucinated_dates(text):
    """
    Safety net for rule 14: local LLMs sometimes ignore instructions and
    write out a calendar date anyway (e.g. "on 13th September"), which can
    be wrong even when the underlying data is correct. Since the response
    is already scoped to the right day via requested_date, any written-out
    date is redundant at best and wrong at worst -- so remove it rather
    than trust it.
    """
    # "13th September", "September 13", "13 September 2026", with optional year
    pattern = re.compile(
        rf"\b(?:on\s+)?(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTH_NAMES})|"
        rf"(?:{_MONTH_NAMES})\s+\d{{1,2}}(?:st|nd|rd|th)?)"
        rf"(?:,?\s+\d{{4}})?\b",
        flags=re.I,
    )
    cleaned = pattern.sub("", text)
    # collapse any double spaces / stray punctuation left behind
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;])\1+", r"\1", cleaned)  # e.g. ",," -> ","
    cleaned = re.sub(r",\s*\.", ".", cleaned)         # e.g. ", ." -> "."
    return cleaned.strip()


# -----------------------------------------------------------------------------
# Main public function
# -----------------------------------------------------------------------------


def get_weather(latitude, longitude, query):
    decoded = question_decoder(query)

    specified = check_place(query)
    location = None

    if specified:
        location = get_cords(specified["location_name"])
        if not location:
            raise ValueError(f"Could not find the location: {specified['location_name']}")
        latitude = location["latitude"]
        longitude = location["longitude"]
    else:
        location = get_place_name(latitude, longitude)

    debug_log("resolved location", {"specified": specified, "latitude": latitude, "longitude": longitude})

    raw = fetch_weather(latitude, longitude)
    weather = normalize_weather(raw)

    # Resolve relative dates against the forecast location's own local clock.
    # This prevents the machine running WeatherGPT from shifting "tomorrow"
    # because of timezone/date differences.
    time_reference = decoded.get("time_reference", "none")
    forecast_current_time = raw.get("current", {}).get("time")
    target_date = resolve_target_date(
        query,
        time_reference,
        raw.get("timezone"),
        forecast_current_time,
    )
    decoded["target_date"] = target_date
    selected = select_weather_for_query(weather, target_date, time_reference)

    selected["location"] = {
        "city": location.get("city") or location.get("name"),
        "state": location.get("state"),
        "country": location.get("country"),
        "latitude": raw["latitude"],
        "longitude": raw["longitude"],
        "timezone": raw["timezone"],
    }

    selected["warnings"] = get_warnings(selected)
    return selected, decoded


if __name__ == "__main__":
    # Basic offline self-test for the deterministic date/time-selection layer.
    fake = {
        "current": {"temperature": 99},
        "hourly": [
            {"time": "2026-09-12T09:00", "temperature": 1},
            {"time": "2026-09-13T07:00", "temperature": 2},
            {"time": "2026-09-13T19:00", "temperature": 3},
            {"time": "2026-09-14T07:00", "temperature": 4},
        ],
        "daily": [
            {"time": "2026-09-12", "temperature_max": 10},
            {"time": "2026-09-13", "temperature_max": 20},
            {"time": "2026-09-14", "temperature_max": 30},
        ],
        "location": {},
        "source": "Open-Meteo",
    }
    result = select_weather_for_query(fake, "2026-09-13", "tomorrow")
    assert result["daily"][0]["time"] == "2026-09-13"
    assert all(x["time"].startswith("2026-09-13") for x in result["hourly"])
    assert result["current"] is None
    print("SELF-TEST PASSED")

    # Quick sanity check for the <think>-stripping fix:
    sample = "<think>reasoning here</think>\n```json\n{\"a\": 1}\n```"
    cleaned = clean_json_text(sample)
    assert cleaned == '{"a": 1}', cleaned
    print("CLEAN_JSON_TEXT TEST PASSED")
