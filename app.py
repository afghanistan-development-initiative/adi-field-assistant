"""
ZaminAI API v4 — Fixes: Gemini response, better error handling
"""
import os, json, logging, requests
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
app = Flask(__name__)
CORS(app, origins="*")

# ── GEE ──────────────────────────────────────────────────────────────────────
gee_ok = False
try:
    import ee
    sa  = os.environ.get("GEE_SERVICE_ACCOUNT","")
    key = os.environ.get("GEE_PRIVATE_KEY","").replace("\\n","\n")
    if sa and key:
        ee.Initialize(ee.ServiceAccountCredentials(sa, key_data=key))
        gee_ok = True
        log.info("GEE OK")
except Exception as e:
    log.error(f"GEE: {e}")

# ── GEMINI REST ───────────────────────────────────────────────────────────────
GEMINI_KEY = os.environ.get("GEMINI_API_KEY","")

def ask_gemini(question, system_ctx):
    """Call Gemini REST API with proper safety settings."""
    if not GEMINI_KEY:
        return None
    # Try flash first, then pro
    for model in ["gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-pro"]:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"
            payload = {
                "contents": [{
                    "parts": [{"text": f"{system_ctx}\n\nQuestion: {question}"}]
                }],
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 300
                }
            }
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code == 200:
                d = r.json()
                # Handle blocked responses
                candidates = d.get("candidates", [])
                if not candidates:
                    log.warning(f"Gemini {model}: no candidates - {d.get('promptFeedback','')}")
                    continue
                c = candidates[0]
                # Check finish reason
                finish = c.get("finishReason","")
                if finish == "SAFETY":
                    log.warning(f"Gemini {model}: blocked by safety")
                    continue
                content = c.get("content",{}).get("parts",[{}])[0].get("text","")
                if content:
                    log.info(f"Gemini {model}: OK ({len(content)} chars)")
                    return content.strip()
            else:
                log.warning(f"Gemini {model}: HTTP {r.status_code} - {r.text[:100]}")
        except Exception as e:
            log.error(f"Gemini {model} error: {e}")
    return None

# ── ANTHROPIC FALLBACK ────────────────────────────────────────────────────────
ANT_KEY = os.environ.get("ANTHROPIC_API_KEY","")

def ask_anthropic(question, system_ctx):
    if not ANT_KEY:
        return None
    try:
        r = requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key":ANT_KEY,"anthropic-version":"2023-06-01","content-type":"application/json"},
            json={"model":"claude-haiku-4-5-20251001","max_tokens":300,
                  "system":system_ctx,"messages":[{"role":"user","content":question}]},
            timeout=15)
        if r.status_code == 200:
            return r.json()["content"][0]["text"].strip()
        log.warning(f"Anthropic HTTP {r.status_code}")
    except Exception as e:
        log.error(f"Anthropic: {e}")
    return None

# ── REGIONAL DB ───────────────────────────────────────────────────────────────
PROVINCES = [
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
    (32.0,33.5,67.0,68.5,"Ghazni",   0.145,-0.21,185,{2019:0.157,2020:0.151,2021:0.139,2022:0.112,2023:0.123,2024:0.138,2025:0.144}),
    (34.5,35.5,67.0,68.5,"Bamyan",   0.152,-0.18,270,{2019:0.164,2020:0.158,2021:0.146,2022:0.119,2023:0.131,2024:0.145,2025:0.151}),
    (33.0,34.0,69.0,70.5,"Logar",    0.148,-0.20,260,{2019:0.159,2020:0.153,2021:0.141,2022:0.114,2023:0.126,2024:0.141,2025:0.147}),
    (32.5,33.5,68.0,69.5,"Paktia",   0.155,-0.18,285,{2019:0.167,2020:0.161,2021:0.149,2022:0.122,2023:0.134,2024:0.149,2025:0.154}),
]

def get_province(lat, lon):
    for a,b,d,e,name,ndvi,water,rain,trend in PROVINCES:
        if a<=lat<=b and d<=lon<=e:
            return name,ndvi,water,rain,trend
    # Default based on rough lat/lon
    ndvi = round(max(0.10, min(0.42, 0.08+lat*0.003+(lon-62)*0.002)), 4)
    rain = max(100, min(450, int(lat*8)))
    return "Afghanistan",ndvi,-0.19,rain,{2019:ndvi+0.015,2020:ndvi+0.010,2021:ndvi+0.003,2022:ndvi-0.030,2023:ndvi-0.012,2024:ndvi,2025:ndvi+0.008}

def calc_area(coords):
    import math
    n=len(coords); a=0
    for i in range(n):
        j=(i+1)%n
        dx=(coords[j][1]-coords[i][1])*111320*math.cos(math.radians((coords[i][0]+coords[j][0])/2))
        dy=(coords[j][0]-coords[i][0])*111320
        a+=coords[i][0]*111320*dx - coords[i][1]*111320*math.cos(math.radians(coords[i][0]))*dy
    return round(abs(a)/2/10000, 2)

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status":"ok","gee":gee_ok,
        "ai":"gemini" if GEMINI_KEY else ("anthropic" if ANT_KEY else "none"),
        "version":"4.0"})

@app.route("/analyse", methods=["POST","OPTIONS"])
def analyse():
    if request.method=="OPTIONS": return jsonify({}),200
    try:
        data=request.get_json(force=True)
        coords=data.get("coords",[])
        year=int(data.get("year",2024))
        label=data.get("label","Field")
        if len(coords)<3: return jsonify({"error":"Need ≥3 points"}),400
        lats=[c[0] for c in coords]; lons=[c[1] for c in coords]
        clat=sum(lats)/len(lats); clon=sum(lons)/len(lons)
        area_ha=calc_area(coords); area_jereb=round(area_ha*5,1)
        if gee_ok:
            try:
                r=_gee_analyse(coords,year,clat,clon)
                r.update({"label":label,"area_ha":area_ha,"area_jereb":area_jereb,"status":"success"})
                return jsonify(r)
            except Exception as e:
                log.error(f"GEE: {e}")
        prov,ndvi,water,rain,trend=get_province(clat,clon)
        return jsonify({"label":label,"ndvi":ndvi,"mndwi":water,"water":water,"rain":rain,
            "area_ha":area_ha,"area_jereb":area_jereb,"province":prov,"trend":trend,
            "ndvi_trend":trend,"year":year,"latest_date":f"{year}-05-15",
            "image_date":f"{year}-05","source":"regional","status":"success",
            "lat":round(clat,5),"lon":round(clon,5)})
    except Exception as e:
        return jsonify({"error":str(e)}),500

def _gee_analyse(coords, year, clat, clon):
    import ee
    poly=ee.Geometry.Polygon([[[c[1],c[0]] for c in coords]])
    ed=f"{year}-07-31" if year<2025 else "2025-05-31"
    col=(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
         .filterBounds(poly).filterDate(f"{year}-04-01",ed)
         .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",20))
         .sort("CLOUDY_PIXEL_PERCENTAGE").limit(5).median().clip(poly))
    ndvi=col.normalizedDifference(["B8","B4"])
    mndwi=col.normalizedDifference(["B3","B11"])
    def mean(img,b):
        v=img.reduceRegion(ee.Reducer.mean(),poly,10,maxPixels=1e8).get(b).getInfo()
        return round(float(v),4) if v else None
    nv=mean(ndvi,"nd"); mv=mean(mndwi,"nd")
    rain=(ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
          .filterBounds(poly).filterDate(f"{year}-01-01",f"{year}-12-31")
          .select("precipitation").sum().clip(poly))
    rv=mean(rain,"precipitation")
    trend={}
    for yr in [2019,2020,2021,2022,2023,2024]:
        try:
            c2=(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(poly).filterDate(f"{yr}-05-01",f"{yr}-07-31")
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",25))
                .median().clip(poly))
            v=c2.normalizedDifference(["B8","B4"]).reduceRegion(ee.Reducer.mean(),poly,10,maxPixels=1e8).get("nd").getInfo()
            trend[yr]=round(float(v),4) if v else None
        except: trend[yr]=None
    return {"ndvi":nv,"mndwi":mv,"water":mv,"rain":rv,"trend":trend,"ndvi_trend":trend,
            "image_date":f"{year}-05","latest_date":f"{year}-05","lat":round(clat,5),"lon":round(clon,5)}

@app.route("/ask", methods=["POST","OPTIONS"])
def ask():
    if request.method=="OPTIONS": return jsonify({}),200
    try:
        data=request.get_json(force=True)
        q=data.get("question","")
        lang=data.get("language","en")
        ctx=data.get("context","")
        fd=data.get("field_data",{})
        if not q: return jsonify({"error":"No question"}),400
        # Build context
        if isinstance(fd,dict) and fd:
            ctx=f"Field: NDVI={fd.get('ndvi','?')}, Water={fd.get('mndwi',fd.get('water','?'))}, Rain={fd.get('rain','?')}mm, Area={fd.get('area_ha','?')}ha, Province={fd.get('province','Afghanistan')}"
        elif not ctx:
            ctx="No field data. Give general farming advice for Afghanistan."
        lang_inst={"fa":"Respond in Afghan Dari (دری). Use دهقان for farmer, جریب for land, تخم for seeds.","ps":"Respond in Pashto (پښتو). Use proper Pashto farming terms.","en":"Respond in English."}.get(lang,"Respond in English.")
        system=f"""You are ZaminAI, an expert agricultural advisor for Afghan smallholder farmers.
Satellite field data: {ctx}
Rules: {lang_inst} Be specific with amounts (kg/jereb, AFN costs, days). Keep answer under 80 words. Never mention AI or satellites."""
        reply = ask_gemini(q, system) or ask_anthropic(q, system)
        if not reply:
            msgs={"fa":"متأسفانه پاسخ دریافت نشد. لطفاً دوباره بپرسید.","ps":"معذرت غواړم، ځواب نه دی. بیا وپوښتئ.","en":"Sorry, no response received. Please try again."}
            reply=msgs.get(lang,msgs["en"])
        return jsonify({"reply":reply,"answer":reply,"model":"gemini" if GEMINI_KEY else "anthropic"})
    except Exception as e:
        log.error(f"Ask: {e}")
        return jsonify({"error":str(e)}),500

@app.route("/ndvi_tile", methods=["POST","OPTIONS"])
def ndvi_tile():
    if request.method=="OPTIONS": return jsonify({}),200
    if not gee_ok: return jsonify({"status":"error"}),503
    try:
        import ee
        data=request.get_json(force=True)
        coords=data.get("coords",[]); year=int(data.get("year",2024))
        poly=ee.Geometry.Polygon([[[c[1],c[0]] for c in coords]])
        ed=f"{year}-07-31" if year<2025 else "2025-05-31"
        col=(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
             .filterBounds(poly).filterDate(f"{year}-04-01",ed)
             .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",20)).median().clip(poly))
        ndvi=col.normalizedDifference(["B8","B4"])
        url=ndvi.getThumbURL({"min":0,"max":0.7,"palette":["#d73027","#fc8d59","#fee08b","#d9ef8b","#91cf60","#1a9850"],"dimensions":512,"format":"png"})
        return jsonify({"status":"success","tile_url":url})
    except Exception as e:
        return jsonify({"status":"error","error":str(e)}),500

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)),debug=False)
