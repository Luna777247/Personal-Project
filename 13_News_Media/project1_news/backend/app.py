from newspaper import Article
import asyncio
import json
from fastapi import FastAPI, WebSocket
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(crawl())
    yield


app = FastAPI(lifespan=lifespan)

# Route GET /
@app.get("/")
async def root():
    return {"message": "Welcome to the News WebSocket API!"}

# Route GET /health
@app.get("/health")
async def health():
    return {"status": "ok"}

# Route POST /echo
from fastapi import Request
@app.post("/echo")
async def echo(request: Request):
    data = await request.json()
    return {"echo": data}

clients = set()

URLS = [
    "https://www.thehindu.com/news/cities/Madurai/",
    "https://timesofindia.indiatimes.com/city/"
]

async def crawl():
    seen_urls = set()
    while True:
        for root_url in URLS:
            try:
                article = Article(root_url)
                article.download()
                article.parse()
                if article.url and article.url not in seen_urls:
                    seen_urls.add(article.url)
                    data = {
                        "title": article.title,
                        "url": article.url,
                        "description": article.summary if hasattr(article, 'summary') else '',
                        "publishedAt": str(article.publish_date) if article.publish_date else '',
                    }
                    for ws in clients:
                        await ws.send_text(json.dumps(data))
            except Exception as e:
                print("Error:", e)
        await asyncio.sleep(60)  # crawl lại sau mỗi 1 phút


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except:
        clients.remove(ws)
