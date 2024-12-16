import asyncio
import base64
import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.templating import Jinja2Templates
from starlette.responses import StreamingResponse
from starlette.staticfiles import StaticFiles

from loader import load

app = FastAPI(docs_url="/docs", redoc_url=None, static_url_path="/static")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
depth_anything = load("vits")
cache = None


def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


@app.get("/")
async def index():
    return templates.TemplateResponse("index.html", {"request": {}})


@app.post("/")
async def handle_form(file: UploadFile = File(...), N: int = Form(10), action: str = Form(...)):
    raw_image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    maxsize = 2000
    if raw_image.shape[0] > maxsize or raw_image.shape[1] > maxsize:
        largest_side = max(raw_image.shape[:2])
        # scale the larger size to 2000 and the other size proportionally
        scale_factor = maxsize / largest_side
        raw_image = cv2.resize(raw_image, (int(raw_image.shape[1] * scale_factor), int(raw_image.shape[0] * scale_factor)))

    cv2.imwrite(f"uploads/{file.filename}", raw_image)
    depth = depth_anything.infer_image(raw_image)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_val = depth.flatten()
    depth_val = depth_val[depth_val > 0]
    # depth_val = depth_val[depth_val < 255]
    depth_val = np.sort(depth_val)
    if action == "depthmap":
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        _, buffer = cv2.imencode('.png', depth.astype(np.uint8))
        return StreamingResponse(io.BytesIO(buffer), media_type="image/png")

    quantiles = [int(depth_val[int(i * len(depth_val) / N)]) for i in range(N)] + [255]
    tolerance = np.ceil(N / 10)
    if action == "parallax":
        depths = []
        layers = {}
        for i in range(len(quantiles) - 1):
            lq, hq = quantiles[i], quantiles[i + 1]
            mask = (depth >= lq - tolerance) & (depth <= hq + tolerance)
            mask = np.repeat(mask[..., np.newaxis], 3, axis=-1)
            masked_image = raw_image * mask
            masked_image = np.concatenate([masked_image, mask[:, :, 0:1] * 255], axis=2)
            depths.append(lq + hq // 2)
            layers[hq] = encode_image_to_base64(masked_image)

        # value that must be over 255. shift is defined by 1/(bg-height)
        bg = 300
        layers[bg] = encode_image_to_base64(cv2.GaussianBlur(raw_image, (5, 5), 0))
        depths = sorted(layers.keys())
        return templates.TemplateResponse("parallax.html", {"request": {}, "layers": layers, "depths": depths, "bg": bg})

    elif action == "histogram":
        plt.hist(depth_val, bins=256)
        # add vertical lines for the quantiles
        for q in quantiles:
            plt.axvline(q, color='r', linestyle='dashed', linewidth=1)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='jpg')
        plt.close()
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/jpg")

    else:
        raise HTTPException(status_code=400, detail="Invalid action specified.")


@app.get("/monalisa")
def monalisa_parallax():
    global cache
    if cache is None:
        with open("static/monalisa.jpg", "rb") as f:
            file = UploadFile(f, filename="static/monalisa.jpg")
            cache = asyncio.run(handle_form(file, 20, "parallax"))
    return cache


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=3000)
