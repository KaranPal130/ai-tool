import os
import requests
import shutil
import time
from fastapi import UploadFile, FastAPI, WebSocket
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from bardapi import Bard
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from utils import hashtags
from config import CAPTION_URL, BARD_KEY


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*","https://a-imagery.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xxl")

os.environ["_BARD_API_KEY"] =  BARD_KEY # type: ignore
# os.environ['CAPTION_URL'] = CAPTION_URL # type: ignore


class Caption(BaseModel):
    caption: str


class EmotionParameter(BaseModel):
    image_description: str
    emotion: str

class Prompt(BaseModel):
    prompt: str

async def send_notification():
    print("Sending a notification before starting the FastAPI server......")


@app.websocket("/notify")
async def notify(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Server is ready to use")


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

    caption = predict_step([destination])
    caption2 = creatingCaption(caption)
    hashtag = hashtags(caption)
    print("--- %s seconds ---" % (time.time() - start_time))
    return {
        'image-description': caption,
        'hash-tags': hashtag,
        'captions': caption2
    }

@app.post("/prompt")
async def prompting(prompt: Prompt):
    url = CAPTION_URL
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json={"prompt": f'{prompt}'}) # type: ignore
    if response.status_code == 200:
        data = response.json()
        print('data', data)
        return data
    else:
        return "Error:", response.status_code
    

def predict_step(image_paths):
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            images.append(i_image)

        # unconditional image captioning
        inputs = processor(images[0], return_tensors="pt")
        out = model.generate(**inputs, num_beams=5, max_new_tokens=30, # type: ignore
                             repetition_penalty=1.0, length_penalty=1.0, temperature=1)  
        return processor.decode(out[0], skip_special_tokens=True)

def creatingCaption(caption):
    return Bard().get_answer(str(f'Create a few captions with emojis and hashtags for the {caption}'))['content']
    url = CAPTION_URL
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json={"prompt": f'Create a few captions with emojis and hashtags for the {caption}'}) # type: ignore
    if response.status_code == 200:
        data = response.json()
        print('data', data)
        return data
    else:
        return "Error:", response.status_code



@app.post("/poetic")
def poeticCaption(caption: Caption):
    return Bard().get_answer(str(f'Make this caption poetic {caption.caption}'))['content']
    


@app.post("/quote")
def createQuote(caption: Caption):
    return Bard().get_answer(str(f'Find a few quotes related to this caption {caption.caption}'))['content']


@app.post("/emotion")
def emotionCaption(emotion: EmotionParameter):
    return Bard().get_answer(str(f'Create a few captions with emojis and hashtags for the {emotion.image_description}, make sure that the captions convey {emotion.emotion} emotion'))['content']
    # url = CAPTION_URL
    # headers = {
    #     "Content-Type": "application/json"
    # }
    # response = requests.post(url, headers=headers, json={"prompt": f'Create a few captions with emojis and hashtags for the {emotion.image_description}, make sure that the captions convey {emotion.emotion} emotion'}) # type: ignore
    # if response.status_code == 200:
    #     data = response.json()
    #     return {
    #         'emotion_caption': data
    #     }
    # else:
    #     return "Error:", response.status_code


@app.post('/questions')
def populateQuestions(caption: Caption):
    return Bard().get_answer(str(f'Give some interesting question on {caption.caption}'))['content']
    # url = CAPTION_URL
    # headers = {
    #     "Content-Type": "application/json"
    # }
    # response = requests.post(url, headers=headers, json={"prompt": f'Give some interesting question on {caption.caption}'}) # type: ignore
    # if response.status_code == 200:
    #     data = response.json()
    #     return {
    #         'emotion_caption': data
    #     }
    # else:
    #     return "Error:", response.status_code


@app.post("/recheckCaption")
def recheckCaption(file: UploadFile):
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

    image = open(destination, "rb").read()
    bard_answer = Bard().ask_about_image(
        'Create a few captions with emojis and hashtags for the', image)
    print(bard_answer['content'])
    caption2 = creatingCaption(bard_answer['content'])
    return {
        'caption': bard_answer['content']
    }
