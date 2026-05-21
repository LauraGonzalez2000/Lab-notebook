# %% [markdown]
# # Visualize Raw Data

# %%
# general python modules for scientific analysis
import os, pathlib
from PIL import Image
import os
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

#%%
#For the 5/6 images

# Paths
NI_FOLDER = os.path.join(str(pathlib.Path(__file__).resolve().parents[1]),'src', 'physion', 'visual_stim', 'NI_bank')

#%%
#Similarity metric between 2 images 
# Image 1 vs im2

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

path1 = os.path.join(NI_FOLDER, "1.jpg")
path2 = os.path.join(NI_FOLDER, "2.jpg")
# Load images
img1 = Image.open(path1).convert("L").convert("RGB")
img2 = Image.open(path2).convert("L").convert("RGB")

plt.imshow(img1)
plt.title("Image 1")
plt.axis('off')
plt.show()

plt.imshow(img2)
plt.title("Image 2")
plt.axis('off')
plt.show()

# Preprocess
inputs = processor(images=[img1, img2], return_tensors="pt")
# Forward pass
with torch.no_grad():
    outputs = model.vision_model(pixel_values=inputs["pixel_values"])
    features = outputs.pooler_output
# Normalize
features = F.normalize(features, p=2, dim=1)
# Cosine similarity
similarity = (features[0] @ features[1]).item()

print("Similarity:", similarity)

#%%
for i in range(1,7):
    path = os.path.join(NI_FOLDER, f"{i}.jpg")
    img = Image.open(path).convert("L").convert("RGB")
    plt.imshow(img)
    plt.axis('off')
    plt.show()
#%% Matrix

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Folder with images
image_files = [f"{i}.jpg" for i in range(1, 7)]
paths = [os.path.join(NI_FOLDER, f) for f in image_files]

# Load images
images = [Image.open(p).convert("L").convert("RGB") for p in paths]

# Preprocess all at once
inputs = processor(images=images, return_tensors="pt")

# Get embeddings
with torch.no_grad():
    outputs = model.vision_model(pixel_values=inputs["pixel_values"])
    embeddings = outputs.pooler_output

# Normalize
embeddings = F.normalize(embeddings, p=2, dim=1)

# Similarity matrix (cosine similarity)
sim_matrix = embeddings @ embeddings.T

print(sim_matrix)

# Heatmap
# convert to numpy for plotting
sim_matrix = sim_matrix.cpu().numpy()

plt.figure(figsize=(6, 5))
plt.imshow(sim_matrix, vmin=0, vmax=1)
plt.colorbar(label="Cosine similarity")

plt.xticks(range(6), [1,2,3,4,5,6])
plt.yticks(range(6), [1,2,3,4,5,6])

plt.title("CLIP Representational Similarity Matrix")
plt.tight_layout()
plt.show()



###############################################
#%%
###############################################
#For the 10 images

NI_FOLDER = 'C:\\Users\\laura.gonzalez\\Desktop\\NI_bank'

for i in range(1,12):
    path = os.path.join(NI_FOLDER, f"{i}.png")
    img = Image.open(path).convert("L").convert("RGB")
    plt.imshow(img)
    plt.axis('off')
    plt.show()

#%% Matrix

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Folder with images
image_files = [f"{i}.png" for i in range(1, 12)]
paths = [os.path.join(NI_FOLDER, f) for f in image_files]

# Load images
images = [Image.open(p).convert("L").convert("RGB") for p in paths]

# Preprocess all at once
inputs = processor(images=images, return_tensors="pt")

# Get embeddings
with torch.no_grad():
    outputs = model.vision_model(pixel_values=inputs["pixel_values"])
    embeddings = outputs.pooler_output   # shape: [5, D]

# Normalize
embeddings = F.normalize(embeddings, p=2, dim=1)

# Similarity matrix (cosine similarity)
sim_matrix = embeddings @ embeddings.T

print(sim_matrix)

# Heatmap
# convert to numpy for plotting
sim_matrix = sim_matrix.cpu().numpy()

plt.figure(figsize=(6, 5))
plt.imshow(sim_matrix, vmin=0, vmax=1)
plt.colorbar(label="Cosine similarity")

plt.xticks(range(11), [1,2,3,4,5,6, 7, 8, 9, 10, 11])
plt.yticks(range(11), [1,2,3,4,5,6, 7, 8, 9, 10, 11])

plt.title("CLIP Representational Similarity Matrix")
plt.tight_layout()
plt.show()
#%%