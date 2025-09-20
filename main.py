from fastapi import FastAPI

app = FastAPI()

@app.post("/")
async def index():
    return {"msg": "ew"}