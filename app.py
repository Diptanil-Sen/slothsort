import json, os, re, shutil
from pathlib import Path
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
LOG = ".slothsort-undo.json"
TEXT = {".txt", ".md", ".csv", ".json", ".py", ".log", ".html"}
KINDS = {"Images": {".jpg", ".jpeg", ".png", ".gif", ".webp"}, "Documents": {".pdf", ".doc", ".docx", ".odt"},
         "Spreadsheets": {".xls", ".xlsx", ".csv"}, "Archives": {".zip", ".tar", ".gz", ".7z"},
         "Installers": {".deb", ".AppImage", ".exe"}, "Media": {".mp3", ".mp4", ".mkv", ".wav"}}
app = FastAPI(title="SlothSort")

class Scan(BaseModel):
    folder: str
class Move(BaseModel):
    src: str
    folder: str
    name: str
class Apply(BaseModel):
    folder: str
    moves: list[Move]

PROMPT = """You tidy a messy folder. For each file below choose a short folder name (like Invoices, Photos, School,
Code, Music, Installers, Documents) and a clear descriptive file name WITHOUT the extension (lowercase words joined
by hyphens; include a date if one appears in the name or text). Reuse the same folder names for similar files.
Return ONLY a JSON array, no markdown: [{{"id":1,"folder":"...","name":"..."}}]
Files:
{files}"""

def root(p):
    d = Path(p).expanduser().resolve()
    if not d.is_dir():
        raise HTTPException(400, "That folder doesn't exist.")
    return d

def clean(s, fallback):
    s = re.sub(r"[^\w .\-]", "", str(s)).strip(" .")[:60]
    return s or fallback

def kind(ext):
    return next((k for k, v in KINDS.items() if ext.lower() in {e.lower() for e in v}), "Other")

async def pick_model(c):
    try:
        r = await c.get(f"{OLLAMA}/api/tags")
    except httpx.HTTPError:
        raise HTTPException(502, "Can't reach Ollama. Make sure it is running.")
    names = [m["name"] for m in r.json().get("models", []) if "gemma" in m["name"]]
    if not names:
        raise HTTPException(502, "No Gemma model found. Run: ollama pull gemma3:4b")
    return os.getenv("GEMMA_MODEL") or names[0]

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/api/scan")
async def scan(s: Scan):
    d = root(s.folder)
    files = sorted(f for f in d.iterdir() if f.is_file() and not f.name.startswith("."))[:40]
    if not files:
        raise HTTPException(400, "No loose files to tidy in that folder.")
    lines = []
    for i, f in enumerate(files, 1):
        hint = ""
        if f.suffix.lower() in TEXT:
            hint = " | starts: " + f.read_text(errors="ignore")[:300].replace("\n", " ")
        lines.append(f"{i}. {f.name}{hint}")
    try:
        async with httpx.AsyncClient(timeout=900) as c:
            body = {"model": await pick_model(c), "temperature": 0.2,
                    "messages": [{"role": "user", "content": PROMPT.format(files="\n".join(lines))}]}
            r = await c.post(f"{OLLAMA}/v1/chat/completions", json=body)
    except httpx.HTTPError:
        raise HTTPException(502, "The model did not answer. Is Ollama running?")
    m = re.search(r"\[.*\]", r.json()["choices"][0]["message"]["content"], re.S)
    try:
        ideas = {int(x["id"]): x for x in json.loads(m.group(0))}
    except Exception:
        ideas = {}
    items = []
    for i, f in enumerate(files, 1):
        x = ideas.get(i, {})
        items.append({"src": f.name, "folder": clean(x.get("folder"), kind(f.suffix)),
                      "name": clean(x.get("name"), f.stem) + f.suffix})
    return {"folder": str(d), "items": items}

@app.post("/api/apply")
def apply(a: Apply):
    d, log, n = root(a.folder), [], 0
    for m in a.moves:
        src = d / m.src
        if src.parent != d or not src.is_file():
            continue
        try:
            dest_dir = d / clean(m.folder, "Other")
            dest_dir.mkdir(exist_ok=True)
            stem = clean(Path(m.name).stem, src.stem)
            dest, k = dest_dir / (stem + src.suffix), 2
            while dest.exists():
                dest, k = dest_dir / f"{stem}-{k}{src.suffix}", k + 1
            shutil.move(str(src), str(dest))
            log.append({"from": str(src), "to": str(dest)})
            n += 1
        except OSError:
            continue
    (d / LOG).write_text(json.dumps(log))
    return {"moved": n}

@app.post("/api/undo")
def undo(s: Scan):
    d = root(s.folder)
    lp = d / LOG
    if not lp.exists():
        raise HTTPException(400, "Nothing to undo in that folder.")
    n = 0
    for e in reversed(json.loads(lp.read_text())):
        a, b = Path(e["from"]), Path(e["to"])
        if b.exists() and not a.exists():
            shutil.move(str(b), str(a))
            n += 1
            try:
                b.parent.rmdir()
            except OSError:
                pass
    lp.unlink()
    return {"restored": n}
