import os
from re import template
import select
from typing import List
from click import File

from fastapi import FastAPI, File, Request, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from PIL import Image
import requests

from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

app = FastAPI()

# -------------- DATEBASE --------------
# SQLAlchemy setup
DATABASE_URL = "mysql+mysqlconnector://root:1234@localhost:3308/categoreyes"
engine = create_engine(DATABASE_URL)
metadata = MetaData()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Define the images table
images = Table(
    'images', metadata,
    Column('id', Integer, primary_key=True),
    Column('filename', String(255), nullable=False)
)

animal = Table(
    'animal', metadata,
    Column('id', Integer, primary_key=True),
    Column('filename', String(255), nullable=False)
)

etc = Table(
    'etc', metadata,
    Column('id', Integer, primary_key=True),
    Column('filename', String(255), nullable=False)
)

food = Table(
    'food', metadata,
    Column('id', Integer, primary_key=True),
    Column('filename', String(255), nullable=False)
)

human = Table(
    'human', metadata,
    Column('id', Integer, primary_key=True),
    Column('filename', String(255), nullable=False)
)

nature = Table(
    'nature', metadata,
    Column('id', Integer, primary_key=True),
    Column('filename', String(255), nullable=False)
)

place = Table(
    'place', metadata,
    Column('id', Integer, primary_key=True),
    Column('filename', String(255), nullable=False)
)

metadata.create_all(engine)

images_folder = "static/images/upload"
if not os.path.exists(images_folder):
    os.makedirs(images_folder)

# ----------------------------------------


# Mount the static folder to serve images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Use Jinja2Templates for HTML templates
templates = Jinja2Templates(directory="templates")
def get_image_files():
    if os.path.exists(images_folder) and os.path.isdir(images_folder):
        images = [file for file in os.listdir(images_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        return images
    return []

# def get_food_images():
#     images_folder = 'static/images'
#     if os.path.exists(images_folder) and os.path.isdir(images_folder):
#         images = [file for file in os.listdir(images_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')) and file.lower().startswith(('f'))]
#         return images
#     return []

def get_animal_filenames():
    db = engine.connect()
    try:
        query = text("SELECT filename FROM animal")
        result = db.execute(query).fetchall()
        file_names = [row[0] for row in result]
        return file_names
    finally:
        db.close()

def get_food_filenames():
    db = engine.connect()
    try:
        query = text("SELECT filename FROM food")
        result = db.execute(query).fetchall()
        file_names = [row[0] for row in result]
        return file_names
    finally:
        db.close()

def get_human_filenames():
    db = engine.connect()
    try:
        query = text("SELECT filename FROM human")
        result = db.execute(query).fetchall()
        file_names = [row[0] for row in result]
        return file_names
    finally:
        db.close()
        
def get_nature_filenames():
    db = engine.connect()
    try:
        query = text("SELECT filename FROM nature")
        result = db.execute(query).fetchall()
        file_names = [row[0] for row in result]
        return file_names
    finally:
        db.close()

def get_place_filenames():
    db = engine.connect()
    try:
        query = text("SELECT filename FROM place")
        result = db.execute(query).fetchall()
        file_names = [row[0] for row in result]
        return file_names
    finally:
        db.close()

def get_etc_filenames():
    db = engine.connect()
    try:
        query = text("SELECT filename FROM etc")
        result = db.execute(query).fetchall()
        file_names = [row[0] for row in result]
        return file_names
    finally:
        db.close()

def get_all_images():
    if os.path.exists(images_folder) and os.path.isdir(images_folder):
        images = [file for file in os.listdir(images_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        return images
    return []

# def get_images_obj():
#     images_folder = "static/images"

#     # Get a list of all image files in the folder
#     image_files = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]

#     images_obj = {
#             "human": [],
#             "animal": [],
#             "food": [],
#             "nature": [],
#             "place": [],
#             "etc": []
#         }
#     # Loop through each image file
#     for image_file in image_files:
#         # Create the full path to the image
#         image_path = os.path.join(images_folder, image_file)

#         # Open the image
#         image = Image.open(image_path)

#         # Use the processor to prepare the input
#         class_name =[
#                 "a photo of a human",
#                 "a photo of people",
#                 "a photo of an animal",
#                 "a photo of animals",
#                 "a photo of food",
#                 "a photo of nature",
#                 "photo of places and strudtures",
#                 "a photo of documents"
#             ]
#         class_dict = {
#             "a photo of a human": "human",
#             "a photo of people": "human",
#             "a photo of an animal": "animal",
#             "a photo of animals": "animal",
#             "a photo of food": "food",
#             "a photo of nature": "nature",
#             "photo of places and strudtures": "place",
#             "a photo of documents": "documents",
#         }
        
#         inputs = processor(
#             text=class_name,
#             images=image,
#             return_tensors="pt",
#             padding=True
#         )

#         # Assuming 'model' is your pre-trained model
#         outputs = model(**inputs)
#         logits_per_image = outputs.logits_per_image
#         probs = logits_per_image.softmax(dim=1)

#         categoryStr = class_name[probs.argmax()]
#         # max_index = arr.index(max(arr))

#         category = class_dict.get(categoryStr)
#         if (max(probs.tolist()[0]) >= 0.45):
#             category = class_dict.get(categoryStr)
#         else:
#             category = "etc"

#         if category in images_obj:
#             images_obj[category].append(image_file)
        
#     return images_obj

def create_table(category):
    table_name = f"{category}"
    
    # Check if the table already exists in the metadata
    existing_table = metadata.tables.get(table_name)
    
    if existing_table is not None:
        return existing_table
    
    return Table(
        table_name, metadata,
        Column('id', Integer, primary_key=True),
        Column('filename', String(255), nullable=False),
        extend_existing=True  # Allow redefinition
    )

def seperate_category():
    db = SessionLocal()
    try:
        # Query all data from the images table
        result = db.execute(images.select()).fetchall()
        # image_data = [{'id': row.id, 'filename': row.filename} for row in result]
        image_data = [row.filename for row in result]
    finally:
        db.close()

    # Get a list of all image files in the folder
    # image_files = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]
    image_files = [filename for filename in image_data if filename in os.listdir(images_folder)]

    images_obj = {
            "human": [],
            "animal": [],
            "food": [],
            "nature": [],
            "place": [],
            "etc": []
        }
    
    # Use the processor to prepare the input
    class_name =[
            "a photo of a human",
            "a photo of people",
            "a photo of an animal",
            "a photo of animals",
            "a photo of food",
            "a photo of nature",
            "photo of places and strudtures",
            "a photo of documents"
        ]
    class_dict = {
        "a photo of a human": "human",
        "a photo of people": "human",
        "a photo of an animal": "animal",
        "a photo of animals": "animal",
        "a photo of food": "food",
        "a photo of nature": "nature",
        "photo of places and strudtures": "place",
        "a photo of documents": "documents",
    }
    # Loop through each image file
    for image_file in image_files:
        # Create the full path to the image
        image_path = os.path.join(images_folder, image_file)

        # Open the image
        image = Image.open(image_path)

        inputs = processor(
            text=class_name,
            images=image,
            return_tensors="pt",
            padding=True
        )

        # Assuming 'model' is your pre-trained model
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        categoryStr = class_name[probs.argmax()]
        # max_index = arr.index(max(arr))

        category = class_dict.get(categoryStr)
        if (max(probs.tolist()[0]) >= 0.45):
            category = class_dict.get(categoryStr)
        else:
            category = "etc"

        if category in images_obj:
            images_obj[category].append(image_file)

            # Check if the table exists; if not, create it
            table = create_table(category)
            table.create(engine, checkfirst=True)

            # Store the filename in the corresponding database table
            db = SessionLocal()
            try:
                db.execute(table.insert().values(filename=image_file))
                db.commit()
            finally:
                db.close()
    return []

def get_table_names():
    # table_names = metadata.tables.keys()
    table_names = [
        table_name
        for table_name, table in metadata.tables.items()
        if table is not None and SessionLocal().execute(table.select().limit(1)).first() is not None
    ]
    return list(table_names)


@app.get("/", response_class=HTMLResponse)
async def read_main(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

# @app.post("/upload")
# async def create_upload_files(request: Request, files: list[UploadFile] = File(...)):
#     print('업로드 요청')
#     for file in files:
#         file_path = f"static/images/{file.filename}"
#         with open(file_path, "wb") as file_object:
#             file_object.write(file.file.read())

#      # Get the count of files in the static/images folder
#     images_folder = "static/images"
#     file_count = len([f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))])

#     # Prepare context with the file count
#     context = {"request": request, "message": "Files uploaded successfully", "images_count": file_count}
#     return templates.TemplateResponse("gallery_main.html", context)

@app.post("/upload")
async def create_upload_file(request: Request, files: List[UploadFile] = File(...)):
    print(1234)
    # Delete all image files in the 'images' folder
    for filename in os.listdir(images_folder):
        file_path = os.path.join(images_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    db = SessionLocal()
    db.execute(images.delete())
    db.execute(animal.delete())
    db.execute(etc.delete())
    db.execute(food.delete())
    db.execute(human.delete())
    db.execute(nature.delete())
    db.execute(place.delete())
            
    for file in files:
        contents = await file.read()
        file_path = os.path.join(images_folder, file.filename)

        with open(file_path, "wb") as f:
            f.write(contents)

        try:
            db.execute(images.insert().values(filename=file.filename))
            db.commit()
        finally:
            db.close()

    # return JSONResponse(content={"files_uploaded": [file.filename for file in files]})
    context = {"request": request, "message": "Files uploaded successfully"}
    return templates.TemplateResponse("gallery_main.html", context)

@app.get("/gallery/", response_class=HTMLResponse)
async def read_images(request: Request):
    images = get_image_files()
    return templates.TemplateResponse("gallery_main.html", {"request": request, "images": images})

@app.get("/gallery/all", response_class=HTMLResponse)
async def read_all_images(request: Request):
    images = get_all_images()
    return templates.TemplateResponse("gallery_all.html", {"request": request, "images": images})

@app.get("/gallery/animal", response_class=HTMLResponse)
async def read_animal_images(request: Request):
    file_names = get_animal_filenames()
    return templates.TemplateResponse("gallery_animal.html", {"request": request, "file_names": file_names})

@app.get("/gallery/food", response_class=HTMLResponse)
async def read_food_images(request: Request):
    file_names = get_food_filenames()
    return templates.TemplateResponse("gallery_food.html", {"request": request, "file_names": file_names})

@app.get("/gallery/human", response_class=HTMLResponse)
async def read_human_images(request: Request):
    file_names = get_human_filenames()
    return templates.TemplateResponse("gallery_human.html", {"request": request, "file_names": file_names})

@app.get("/gallery/nature", response_class=HTMLResponse)
async def read_nature_images(request: Request):
    file_names = get_nature_filenames()
    return templates.TemplateResponse("gallery_nature.html", {"request": request, "file_names": file_names})

@app.get("/gallery/place", response_class=HTMLResponse)
async def read_place_images(request: Request):
    file_names = get_place_filenames()
    return templates.TemplateResponse("gallery_place.html", {"request": request, "file_names": file_names})

@app.get("/gallery/etc", response_class=HTMLResponse)
async def read_etc_images(request: Request):
    file_names = get_etc_filenames()
    return templates.TemplateResponse("gallery_etc.html", {"request": request, "file_names": file_names})

@app.get("/gallery/origin", response_class=HTMLResponse)
async def read_origin_images(request: Request):
    images = get_all_images()
    return templates.TemplateResponse("gallery_origin.html", {"request": request, "images":images})

# @app.get("/gallery/seperate", response_class=HTMLResponse)
# async def seperate_images(request: Request):
#     images_obj = get_images_obj()
#     print('images_obj :',images_obj)
#     return templates.TemplateResponse("gallery_seperate.html", {"request": request, "images_obj":images_obj})
@app.get("/gallery/seperate", response_class=HTMLResponse)
async def seperate_images(request: Request):
    seperate_category()
    table_names = get_table_names()
    table_names.remove('images')
    return templates.TemplateResponse("gallery_seperate.html", {"request": request, "table_names": table_names})


# image_path = "static/images/h2.jpg"
# image = Image.open(image_path)

# inputs = processor(text=[
#     "a photo of a human",
#     "a photo of a animal", 
#     "a photo of a food",
#     "a photo of a nature",
#     "a photo of a documents"
#     ], images=image, return_tensors="pt", padding=True)

# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image
# probs = logits_per_image.softmax(dim=1)
# print('probs', probs)



# print('serperated_images : ', serperated_images)
    
    
    # # Print the results for the current image
    # print(f"{image_file}의 카테고리: {category}")
    
