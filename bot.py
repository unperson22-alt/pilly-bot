import os
import logging
import asyncio
import urllib.parse
import httpx
from aiohttp import web

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN  = os.environ["TELEGRAM_TOKEN"]
REPLICATE_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")
OFFICE_CHAT_ID  = os.environ.get("OFFICE_CHAT_ID", "")
HTTP_PORT       = int(os.environ.get("PORT", 8080))

TG_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


# ── Image providers ──────────────────────────────────────────────────────────

async def _replicate(prompt: str) -> str | None:
    """Replicate Flux Schnell — основной провайдер."""
    if not REPLICATE_TOKEN:
        return None
    try:
        async with httpx.AsyncClient(timeout=90) as c:
            r = await c.post(
                "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions",
                headers={
                    "Authorization": f"Bearer {REPLICATE_TOKEN}",
                    "Content-Type": "application/json",
                    "Prefer": "wait=60"
                },
                json={"input": {"prompt": prompt, "num_outputs": 1, "output_format": "webp"}}
            )
            d = r.json()
            if d.get("status") == "succeeded":
                logger.info("[pilly] Replicate ✅")
                return d["output"][0]
            if r.status_code == 402 or "insufficient credit" in str(d.get("error", "")):
                logger.warning("[pilly] Replicate: нет кредитов → fallback")
                return None
            logger.error(f"[pilly] Replicate: {d.get('error', d.get('status'))}")
    except Exception as e:
        logger.error(f"[pilly] Replicate exception: {e}")
    return None


async def _pollinations(prompt: str) -> str | None:
    """Pollinations.ai — бесплатный fallback, без токена."""
    try:
        encoded = urllib.parse.quote(prompt)
        url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width=1024&height=1024&nologo=true&enhance=true"
            f"&seed={abs(hash(prompt)) % 99999}"
        )
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.get(url, follow_redirects=True)
            if r.status_code == 200 and "image/" in r.headers.get("content-type", ""):
                logger.info("[pilly] Pollinations ✅")
                return url
    except Exception as e:
        logger.error(f"[pilly] Pollinations exception: {e}")
    return None


async def generate(prompt: str) -> str | None:
    """Replicate → Pollinations fallback."""
    url = await _replicate(prompt)
    if not url:
        logger.info("[pilly] пробую Pollinations...")
        url = await _pollinations(prompt)
    return url


# ── Telegram ─────────────────────────────────────────────────────────────────

async def tg_send_photo(chat_id: int, photo_url: str, caption: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{TG_API}/sendPhoto", json={
                "chat_id": chat_id,
                "photo": photo_url,
                "caption": caption
            })
            if r.status_code == 200:
                return True
            logger.error(f"[pilly] sendPhoto {r.status_code}: {r.text[:200]}")
    except Exception as e:
        logger.error(f"[pilly] sendPhoto exception: {e}")
    return False


async def tg_send_message(chat_id: int, text: str):
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(f"{TG_API}/sendMessage", json={"chat_id": chat_id, "text": text})
    except Exception:
        pass


# ── HTTP endpoints ────────────────────────────────────────────────────────────

async def handle_generate(request: web.Request) -> web.Response:
    """
    POST /generate
    Body: { "prompt": str, "chat_id": int (optional), "requester": str (optional) }
    Generates image and sends it to chat_id (or OFFICE_CHAT_ID as default).
    """
    try:
        body = await request.json()
        prompt    = (body.get("prompt") or "").strip()
        chat_id   = int(body.get("chat_id") or OFFICE_CHAT_ID or 0)
        requester = body.get("requester", "кто-то")

        if not prompt:
            return web.json_response({"status": "error", "message": "empty prompt"}, status=400)
        if not chat_id:
            return web.json_response({"status": "error", "message": "no chat_id"}, status=400)

        logger.info(f"[pilly] /generate from={requester} chat={chat_id} prompt={prompt[:80]}")

        url = await generate(prompt)
        if not url:
            await tg_send_message(chat_id, "❌ Не получилось нарисовать — оба провайдера недоступны")
            return web.json_response({"status": "error", "message": "all providers failed"}, status=500)

        ok = await tg_send_photo(chat_id, url, caption=f"🎨 {prompt}")
        if ok:
            return web.json_response({"status": "ok", "url": url})

        return web.json_response({"status": "error", "message": "failed to send photo"}, status=500)

    except Exception as e:
        logger.error(f"[pilly] /generate exception: {e}")
        return web.json_response({"status": "error", "message": str(e)}, status=500)


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok",
        "name": "pilly-bot",
        "replicate": bool(REPLICATE_TOKEN),
        "office_chat": bool(OFFICE_CHAT_ID),
    })


# ── Startup ───────────────────────────────────────────────────────────────────

async def main():
    app = web.Application()
    app.router.add_post("/generate", handle_generate)
    app.router.add_get("/health",    handle_health)

    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", HTTP_PORT).start()
    logger.info(f"🎨 Pilly запущена на :{HTTP_PORT} | Replicate={'yes' if REPLICATE_TOKEN else 'no'}")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
