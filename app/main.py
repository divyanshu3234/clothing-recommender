from fastapi import FastAPI, UploadFile, File, Form
from app.model import hybrid_recommend
from .model import hybrid_recommend

app = FastAPI(title="Clothing Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "API is running successfully!"}

@app.post("/recommend")
async def recommend_api(
    file: UploadFile = File(...),
    height: float = Form(...),
    weight: float = Form(...)
):

    # Read image
    image_bytes = await file.read()

    # Call model
    results = hybrid_recommend(
        image_bytes,
        [height, weight]
    )

    return {
        "recommendations": results
    }
    
