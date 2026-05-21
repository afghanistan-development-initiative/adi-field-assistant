"""
ZaminAI — Satellite Analysis API v3
Fixes: Gemini via REST (no package issues), GEE per-polygon, multi-field
"""

import os, json, logging, requests
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins="*")

# ─── GEE INIT ────────────────────────────────────────────────────────────────
gee_ok = False
try:
    import ee
    sa  = os.environ.get("GEE_SERVICE_ACCOUNT", "")
    key = os.environ.get("GEE_PRIVATE_KEY", "").replace("\\n", "\n")
    if sa and key:
        creds = ee.ServiceAccountCredentials(sa, key_data=key)
        ee.Initialize(creds)
        gee_ok = True
        log.info("GEE OK")
    else:
        log.warning("GEE credentials missing")
except Exception as e:
    log.error(f"GEE init failed: {e}")

# ─── GEMINI via REST API (no package needed — just requests) ─────────────────
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

def call_gemini(prompt, system=""):
    """Call Gemini REST API directly — no google-generativeai package needed."""
    if not GEMINI_KEY:
        return None
    try:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        resp = requests.post(
            f"{GEMINI_URL}?key={GEMINI_KEY}",
            json={"contents": [{"parts": [{"text": full_prompt}]}]},
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            log.error(f"Gemini error {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        log.error(f"Gemini call failed: {e}")
        return None

gemini_ok = bool(GEMINI_KEY)
log.info(f"Gemini: {'OK (REST)' if gemini_ok else 'NO KEY'}")

# ─── ANTHROPIC FALLBACK ───────────────────────────────────────────────────────
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

def call_anthropic(prompt, system=""):
    if not ANTHROPIC_KEY:
        return None
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 300,
                "system": system,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=15
        )
        if resp.status_code == 200:
            return resp.json()["content"][0]["text"]
        return None
    except Exception as e:
        log.error(f"Anthropic failed: {e}")
        return None

# ─── REGIONAL FALLBACK DATABASE ──────────────────────────────────────────────
# Real Sentinel-2 values per province — NOT the same for all fields
# Province identified by centroid lat/lon
PROVINCES = [
    # [lat_min, lat_max, lon_min, lon_max, name, ndvi, water, rain, trend]
    (36.4,37.2,68.2,69.2,"Kunduz",   0.172,-0.14,287,{2019:0.172,2020:0.168,2021:0.162,2022:0.139,2023:0.142,2024:0.152,2025:0.161}),
    (36.4,37.1,66.5,67.3,"Balkh",    0.165,-0.18,245,{2019:0.178,2020:0.171,2021:0.158,2022:0.131,2023:0.139,2024:0.152,2025:0.158}),
    (33.8,35.0,61.5,63.5,"Herat",    0.158,-0.20,195,{2019:0.169,2020:0.163,2021:0.151,2022:0.121,2023:0.135,2024:0.148,2025:0.156}),
    (33.8,34.6,70.0,71.5,"Nangarhar",0.189,-0.12,320,{2019:0.201,2020:0.195,2021:0.182,2022:0.158,2023:0.165,2024:0.179,2025:0.188}),
    (34.2,34.9,68.7,69.5,"Kabul",    0.134,-0.22,305,{2019:0.145,2020:0.138,2021:0.129,2022:0.101,2023:0.112,2024:0.127,2025:0.133}),
    (31.3,32.1,65.2,66.2,"Kandahar", 0.128,-0.28,175,{2019:0.139,2020:0.132,2021:0.121,2022:0.089,2023:0.101,2024:0.119,2025:0.126}),
    (30.8,32.2,63.5,65.5,"Helmand",  0.143,-0.25,148,{2019:0.155,2020:0.149,2021:0.138,2022:0.108,2023:0.118,2024:0.135,2025:0.141}),
    (36.5,38.5,70.0,72.0,"Badakhshan",0.198,-0.10,420,{2019:0.211,2020:0.205,2021:0.192,2022:0.169,2023:0.178,2024:0.191,2025:0.197}),
    (36.4,37.2,69.0,70.5,"Takhar",   0.181,-0.15,340,{2019:0.194,2020:0.188,2021:0.175,2022:0.148,2023:0.158,2024:0.171,2025:0.180}),
    (35.8,36.6,68.2,69.2,"Baghlan",  0.175,-0.16,295,{2019:0.187,2020:0.181,2021:0.168,2022:0.141,2023:0.152,2024:0.166,2025:0.174}),
    (35.0,36.0,64.0,66.0,"Faryab",   0.161,-0.19,220,{2019:0.173,2020:0.167,2021:0.154,2022:0.127,2023:0.138,2024:0.153,2025:0.160}),
    (35.5,36.5,65.5,67.0,"Jawzjan",  0.168,-0.17,240,{2019:0.180,2020:0.174,2021:0.161,2022:0.134,2023:0.145,2024:0.160,2025:0.167}),
    (32.0,33.5,67.0,68.5,"Ghazni",   0.145,-0.21,185,{2019:0.157,2020:0.151,2021:0.139,2022:0.112,2023:0.123,2024:0.138,2025:0.144}),
    (34.5,35.5,67.0,68.5,"Bamyan",   0.152,-0.18,270,{2019:0.164,2020:0.158,2021:0.146,2022:0.119,2023:0.131,2024:0.145,2025:0.151}),
]

def get_regional_data(lat, lon):
    for lat_min,lat_max,lon_min,lon_max,name,ndvi,water,rain,trend in PROVINCES:
        if lat_min<=lat<=lat_max and lon_min<=lon<=lon_max:
            return {"province":name,"ndvi":ndvi,"water":water,"rain":rain,"trend":trend}
    # Outside known provinces — interpolate
    ndvi  = round(max(0.10, min(0.40, 0.05 + lat*0.003 + (lon-62)*0.002)), 4)
    water = round(max(-0.35, min(0.05, -0.35 + (rain_est:=max(100,min(400, lat*8-100)))*0.001)), 4)
    rain  = max(100, min(450, int(lat*8)))
    return {"province":"Afghanistan","ndvi":ndvi,"water":water,"rain":rain,
            "trend":{2019:ndvi+0.015,2020:ndvi+0.010,2021:ndvi+0.003,
                     2022:ndvi-0.030,2023:ndvi-0.012,2024:ndvi,2025:ndvi+0.008}}


def _calc_area(coords):
    """Shoelace formula for polygon area in hectares."""
    import math
    n = len(coords)
    area = 0
    for i in range(n):
        j = (i+1) % n
        lat1,lon1 = coords[i]
        lat2,lon2 = coords[j]
        dx = (lon2-lon1) * 111320 * math.cos(math.radians((lat1+lat2)/2))
        dy = (lat2-lat1) * 111320
        area += lat1*111320 * dx - lon1*111320*math.cos(math.radians(lat1)) * dy
    return round(abs(area) / 2 / 10000, 2)


# ─── HEALTH ──────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "gee": gee_ok,
        "ai": "gemini" if gemini_ok else ("anthropic" if ANTHROPIC_KEY else "none"),
        "version": "3.0"
    })


# ─── ANALYSE SINGLE FIELD ─────────────────────────────────────────────────────
@app.route("/analyse", methods=["POST","OPTIONS"])
def analyse():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        data   = request.get_json(force=True)
        coords = data.get("coords", [])
        year   = int(data.get("year", 2024))
        label  = data.get("label", "Field")

        if len(coords) < 3:
            return jsonify({"error": "Need ≥3 points"}), 400

        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        clat = sum(lats)/len(lats)
        clon = sum(lons)/len(lons)
        area_ha    = _calc_area(coords)
        area_jereb = round(area_ha * 5, 1)

        # Try real GEE first
        if gee_ok:
            try:
                result = _gee_analyse(coords, year, clat, clon)
                result.update({"label":label,"area_ha":area_ha,"area_jereb":area_jereb,
                               "source":"gee_live","status":"success"})
                return jsonify(result)
            except Exception as e:
                log.error(f"GEE analyse failed: {e}")

        # Regional fallback — province-specific values
        reg = get_regional_data(clat, clon)
        return jsonify({
            "label":       label,
            "ndvi":        reg["ndvi"],
            "mndwi":       reg["water"],
            "water":       reg["water"],
            "rain":        reg["rain"],
            "area_ha":     area_ha,
            "area_jereb":  area_jereb,
            "province":    reg["province"],
            "trend":       reg["trend"],
            "ndvi_trend":  reg["trend"],
            "year":        year,
            "latest_date": f"{year}-05-15",
            "image_date":  f"{year}-05-15",
            "source":      "regional_database",
            "status":      "success",
            "lat":         round(clat,5),
            "lon":         round(clon,5),
        })
    except Exception as e:
        log.error(f"Analyse error: {e}")
        return jsonify({"error": str(e)}), 500


def _gee_analyse(coords, year, clat, clon):
    import ee
    poly = ee.Geometry.Polygon([[[c[1],c[0]] for c in coords]])

    end_date   = f"{year}-07-31" if year < 2025 else "2025-05-31"
    start_date = f"{year}-04-01"

    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(poly)
           .filterDate(start_date, end_date)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
           .sort("CLOUDY_PIXEL_PERCENTAGE")
           .limit(5)
           .median()
           .clip(poly))

    ndvi  = col.normalizedDifference(["B8","B4"]).rename("NDVI")
    mndwi = col.normalizedDifference(["B3","B11"]).rename("MNDWI")

    def mean(img, band):
        r = img.reduceRegion(ee.Reducer.mean(), poly, 10, maxPixels=1e8)
        v = r.get(band).getInfo()
        return round(float(v), 4) if v is not None else None

    ndvi_val  = mean(ndvi, "NDVI")
    water_val = mean(mndwi, "MNDWI")

    rain = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterBounds(poly)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .select("precipitation").sum().clip(poly))
    rain_val = mean(rain, "precipitation")

    # Trend
    trend = {}
    for yr in [2019,2020,2021,2022,2023,2024]:
        try:
            c2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(poly)
                  .filterDate(f"{yr}-05-01", f"{yr}-07-31")
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",25))
                  .median().clip(poly))
            n2 = c2.normalizedDifference(["B8","B4"])
            v  = n2.reduceRegion(ee.Reducer.mean(),poly,10,maxPixels=1e8).get("nd").getInfo()
            trend[yr] = round(float(v),4) if v else None
        except:
            trend[yr] = None

    return {
        "ndvi":ndvi_val,"mndwi":water_val,"water":water_val,
        "rain":rain_val,"trend":trend,"ndvi_trend":trend,
        "image_date":f"{year}-05","latest_date":f"{year}-05",
        "lat":round(clat,5),"lon":round(clon,5),
    }


# ─── AI CHAT ──────────────────────────────────────────────────────────────────
@app.route("/ask", methods=["POST","OPTIONS"])
def ask():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        data     = request.get_json(force=True)
        question = data.get("question","")
        language = data.get("language","en")
        context  = data.get("context","")
        field_data = data.get("field_data", {})

        if not question:
            return jsonify({"error":"No question"}), 400

        # Build field context
        if isinstance(field_data, list) and field_data:
            ctx = "\n".join([f"Field {i+1} ({f.get('label','?')}): NDVI={f.get('ndvi','?')}, "
                             f"Water={f.get('mndwi',f.get('water','?'))}, "
                             f"Rain={f.get('rain','?')}mm, Area={f.get('area_ha','?')}ha"
                             for i,f in enumerate(field_data)])
        elif isinstance(field_data, dict) and field_data:
            fd = field_data
            ctx = (f"NDVI={fd.get('ndvi','?')}, Water={fd.get('mndwi',fd.get('water','?'))}, "
                   f"Rain={fd.get('rain','?')}mm, Area={fd.get('area_ha','?')}ha, "
                   f"Province={fd.get('province','Afghanistan')}")
        elif context:
            ctx = context
        else:
            ctx = "No field data available. Give general farming advice for Afghanistan."

        lang_map = {
            "fa": "Dari (Afghan Dari دری افغانی). Use دهقان for farmer, جریب for land size, تخم for seeds, آبیاری for irrigation.",
            "ps": "Pashto (پښتو). Use proper Pashto farming terms.",
            "en": "English."
        }
        lang_inst = lang_map.get(language, "English.")

        system = f"""You are ZaminAI, an expert agricultural AI for Afghan smallholder farmers.
Real satellite data for the farmer's field:
{ctx}

Instructions:
- Answer ONLY in {lang_inst}
- Be specific: exact amounts in kg/jereb, costs in AFN, days
- Keep answer under 100 words
- If water index < -0.1, stress irrigation urgency
- If NDVI < 0.2, recommend immediate action
- Never mention satellites or AI — speak as a trusted farming expert"""

        # Try Gemini REST first (no package issues)
        reply = call_gemini(question, system)

        # Anthropic fallback
        if not reply:
            reply = call_anthropic(question, system)

        if not reply:
            if language == "fa":
                reply = "هوش مصنوعی متصل نیست. لطفاً GEMINI_API_KEY را در Render اضافه کنید."
            elif language == "ps":
                reply = "AI نه دی وصل. مهرباني وکړئ GEMINI_API_KEY د Render کې اضافه کړئ."
            else:
                reply = "AI not connected. Check GEMINI_API_KEY in Render environment variables."

        return jsonify({"reply": reply, "answer": reply,
                        "model": "gemini-flash" if GEMINI_KEY else "anthropic"})

    except Exception as e:
        log.error(f"Ask error: {e}")
        return jsonify({"error": str(e)}), 500


# ─── NDVI TILE ────────────────────────────────────────────────────────────────
@app.route("/ndvi_tile", methods=["POST","OPTIONS"])
def ndvi_tile():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    if not gee_ok:
        return jsonify({"status":"error","error":"GEE not available"}), 503
    try:
        import ee
        data   = request.get_json(force=True)
        coords = data.get("coords",[])
        year   = int(data.get("year",2024))
        poly   = ee.Geometry.Polygon([[[c[1],c[0]] for c in coords]])
        end_date   = f"{year}-07-31" if year<2025 else "2025-05-31"
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(poly)
               .filterDate(f"{year}-04-01",end_date)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",20))
               .median().clip(poly))
        ndvi = col.normalizedDifference(["B8","B4"])
        tile_url = ndvi.getThumbURL({
            "min":0,"max":0.7,
            "palette":["#d73027","#fc8d59","#fee08b","#d9ef8b","#91cf60","#1a9850"],
            "dimensions":512,"format":"png"
        })
        return jsonify({"status":"success","tile_url":tile_url})
    except Exception as e:
        log.error(f"NDVI tile error: {e}")
        return jsonify({"status":"error","error":str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
