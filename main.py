import uvicorn
from fastapi import FastAPI
from training import training_flow


DATA_PATH = "dataset.csv"
MODEL_PATH = "./"
app = FastAPI()


@app.get("/training")
def training():
    try:
        training_flow(DATA_PATH, MODEL_PATH)
        return {"message": "Training completed successfully."}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)