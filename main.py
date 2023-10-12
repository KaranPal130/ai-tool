import os
import requests
import shutil
import time
from fastapi import UploadFile, FastAPI
from PIL import Image
from transformers import BlipProcessor, Blip2ForConditionalGeneration
from bardapi import Bard
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from utils import hashtags
from config import BARD_KEY

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl")
 
os.environ["_BARD_API_KEY"] = BARD_KEY  # type: ignore


class Caption(BaseModel):
    caption: str


@app.get("/")
async def index():
    return {"message": "Hello World"}


def get_image_url(image_path):
    response = requests.get(image_path)
    return response.url


@app.post("/caption")
async def UploadImage(file: UploadFile):
    start_time = time.time()
    upload_dir = os.path.join(os.getcwd(), "uploads")
    # Create the upload directory if it doesn't exist
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # get the destination path
    destination = os.path.join(upload_dir, file.filename)  # type: ignore
    print(destination)

    # copy the file contents
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    def predict_step(image_paths):
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            images.append(i_image)

        # # conditional image captioning
        # text = "a photography of"
        # inputs = processor(images[0], text, return_tensors="pt")
        #
        # out = model.generate(**inputs)  # type: ignore

        max_length = 4000  # Increase max_length for longer captions
        num_beams = 10
        length_penalty = 2.0  # Adjust length penalty as needed
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "length_penalty": length_penalty}

        # unconditional image captioning
        inputs = processor(images[0], return_tensors="pt")
        out = model.generate(**inputs)  # type: ignore
        return processor.decode(out[0], skip_special_tokens=True)

    caption = predict_step([destination])
    caption2 = creatingCaption(caption)
    hashtag = hashtags(caption)
    print("--- %s seconds ---" % (time.time() - start_time))
    return {
        'image-description': caption,
        'hash-tags': hashtag,
        'captions': caption2
    }


def creatingCaption(caption):
    return Bard().get_answer(str(f'Create a few captions with emojis and hashtags for the {caption}'))['content']


@app.post("/poetic")
def poeticCaption(caption: Caption):
    return Bard().get_answer(str(f'Make this caption poetic {caption.caption}'))['content']


@app.post("/quote")
def createQuote(caption: Caption):
    return Bard().get_answer(str(f'Find a few quotes related to this caption {caption.caption}'))['content']
