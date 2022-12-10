from fastapi import FastAPI
from uvicorn import run as app_run
from fastapi.responses import Response
from src.constants import APP_HOST, APP_PORT
from src.pipeline.tracking import TrackingPipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/track")
async def tracking():
    try:
        tracking_pipeline = TrackingPipeline()

        tracking_pipeline.run_pipeline()

        return Response("Tracked video saved successfully to Gcloud Storage !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
