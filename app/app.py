import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar modelo
model = tf.keras.models.load_model(
    "modelo_perros.keras",
    compile=False
)
# Lista de razas en el mismo orden que se entrenó
razas = [
    'Chihuahua', 'Japanese spaniel', 'Maltese dog', 'Pekinese', 'Shih-Tzu',
    'Blenheim spaniel', 'Papillon', 'Toy terrier', 'Rhodesian ridgeback',
    'Afghan hound', 'Basset', 'Beagle', 'Bloodhound', 'Bluetick',
    'Black-and-tan coonhound', 'Walker hound', 'English foxhound', 'Redbone',
    'Borzoi', 'Irish wolfhound', 'Italian greyhound', 'Whippet', 'Ibizan hound',
    'Norwegian elkhound', 'Otterhound', 'Saluki', 'Scottish deerhound', 'Weimaraner',
    'Staffordshire bullterrier', 'American Staffordshire terrier', 'Bedlington terrier',
    'Border terrier', 'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier',
    'Norwich terrier', 'Yorkshire terrier', 'Wire-haired fox terrier', 'Lakeland terrier',
    'Sealyham terrier', 'Airedale', 'Cairn', 'Australian terrier', 'Dandie Dinmont',
    'Boston bull', 'Miniature schnauzer', 'Giant schnauzer', 'Standard schnauzer',
    'Scotch terrier', 'Tibetan terrier', 'Silky terrier', 'Soft-coated wheaten terrier',
    'West Highland white terrier', 'Lhasa', 'Flat-coated retriever', 'Curly-coated retriever',
    'Golden retriever', 'Labrador retriever', 'Chesapeake Bay retriever',
    'German short-haired pointer', 'Vizsla', 'English setter', 'Irish setter',
    'Gordon setter', 'Brittany spaniel', 'Clumber', 'English springer',
    'Welsh springer spaniel', 'Cocker spaniel', 'Sussex spaniel', 'Irish water spaniel',
    'Kuvasz', 'Schipperke', 'Groenendael', 'Malinois', 'Briard', 'Kelpie',
    'Komondor', 'Old English sheepdog', 'Shetland sheepdog', 'Collie',
    'Border collie', 'Bouvier des Flandres', 'Rottweiler', 'German shepherd',
    'Doberman', 'Miniature pinscher', 'Greater Swiss Mountain dog', 'Bernese mountain dog',
    'Appenzeller', 'EntleBucher', 'Boxer', 'Bull mastiff', 'Tibetan mastiff',
    'French bulldog', 'Great Dane', 'Saint Bernard', 'Eskimo dog', 'Malamute',
    'Siberian husky', 'Affenpinscher', 'Basenji', 'Pug', 'Leonberg',
    'Newfoundland', 'Great Pyrenees', 'Samoyed', 'Pomeranian', 'Chow',
    'Keeshond', 'Brabancon griffon', 'Pembroke', 'Cardigan', 'Toy poodle',
    'Miniature poodle', 'Standard poodle', 'Mexican hairless', 'Dingo',
    'Dhole', 'African hunting dog'
]

st.title("🐶 Clasificador de Razas de Perros")
st.write("Sube una foto de un perro y el modelo intentará identificar su raza.")

imagen_subida = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if imagen_subida is not None:
    img = Image.open(imagen_subida).convert("RGB")
    st.image(img, caption="Imagen cargada", use_column_width=True)

    # Preprocesar igual que en entrenamiento
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    prediccion = model.predict(img_array)
    probabilidades = prediccion[0]

    # Top 5 razas
    top5_idx = np.argsort(probabilidades)[::-1][:5]

    st.subheader("🏆 Raza más probable")
    st.success(f"{razas[top5_idx[0]]} — {probabilidades[top5_idx[0]]*100:.2f}%")

    st.subheader("📊 Top 5 razas")
    for idx in top5_idx:
        st.write(f"{razas[idx]}: {probabilidades[idx]*100:.2f}%")
