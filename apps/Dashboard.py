import os
import socket
import random
import numpy as np
from PIL import Image
import dash
from dash import dcc, html, Output, Input, State
import plotly.express as px
from torchvision import transforms
from torch.utils.data import Dataset

# Define the custom FGVCAircraftDataset class
class FGVCAircraftDataset(Dataset):
    ALLOWED = {"manufacturer", "family", "variant"}

    def __init__(self, root='FGVC/fgvc-aircraft-2013b/data', split="train", level="manufacturer",
                 transform=None, return_class=False, cropped=False, album=False):
        assert level in self.ALLOWED, f"level must be one of {self.ALLOWED}"
        self.root = root
        self.split = split
        self.level = level
        self.transform = transform
        self.return_class = return_class
        self.cropped = cropped
        self.album = album

        class_file = os.path.join(root, f"images_{level}_{split}.txt")
        if not os.path.isfile(class_file):
            raise FileNotFoundError(f"{class_file} not found")

        samples = []
        with open(class_file, "r") as f:
            for ln in f:
                parts = ln.strip().split()
                img_id, *label_parts = parts
                class_str = " ".join(label_parts)
                samples.append((img_id, class_str))
        self.samples = samples

        self.classes = sorted({lbl for _, lbl in samples})
        self.class_to_idx = {lbl: i for i, lbl in enumerate(self.classes)}
        self.idx_to_class = {i: lbl for lbl, i in self.class_to_idx.items()}

        bbox_file = os.path.join(root, "images_box.txt")
        if not os.path.isfile(bbox_file):
            raise FileNotFoundError(f"{bbox_file} not found")

        self.bboxes = {}
        with open(bbox_file, "r") as f:
            for ln in f:
                parts = ln.strip().split()
                img_id = parts[0]
                bbox = tuple(map(int, parts[1:]))
                self.bboxes[img_id] = bbox

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, class_str = self.samples[idx]
        img_path = os.path.join(self.root, "images", f"{img_id}.jpg")
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        if self.cropped:
            xmin, ymin, xmax, ymax = self.bboxes[img_id]
            img = img.crop((xmin, ymin, xmax, ymax))

        if self.transform:
            if not self.album:
                img = self.transform(img)
            else:
                aug = self.transform(image=img_np)
                img = aug['image']

        if self.return_class:
            return img, class_str

        class_idx = self.class_to_idx[class_str]
        return img, class_idx

# Initialize the Dash app
app = dash.Dash(__name__)

LEVELS = ['manufacturer', 'family', 'variant']

app.layout = html.Div([
    html.H2("FGVC Aircraft Dataset Viewer"),
    html.P("Please first download the dataset using torchvision.datasets.FGVCAircraft() "
           "and save it in a directory like 'FGVC/fgvc-aircraft-2013b/data'. Once done, input the full path below."
          "Such as 'C:\\Users\\chihp\\UMich/SIADS/699/FGVC/fgvc-aircraft-2013b/data'"),
    dcc.Input(id='path-input', type='text', placeholder='Enter dataset root path', style={'width': '60%'}),
    html.Button('Load Dataset', id='load-button', n_clicks=0),
    html.Div(id='load-status', style={'marginTop': '10px', 'color': 'green'}),
    html.Br(),
    html.Label("Select Level:"),
    dcc.Dropdown(id='level-dropdown', options=[{'label': lvl.title(), 'value': lvl} for lvl in LEVELS]),
    html.Label("Select Class:"),
    dcc.Dropdown(id='class-dropdown'),
    html.Div(id='image-output', style={'display': 'flex', 'flexWrap': 'wrap'})
])

# Store loaded datasets in memory
loaded_datasets = {}

@app.callback(
    Output('load-status', 'children'),
    Output('level-dropdown', 'value'),
    Input('load-button', 'n_clicks'),
    State('path-input', 'value')
)
def load_dataset(n_clicks, path):
    if n_clicks > 0 and path:
        try:
            for level in LEVELS:
                loaded_datasets[level] = FGVCAircraftDataset(root=path, split='train', level=level, return_class=True)
            return f"Dataset loaded successfully from: {path}", 'manufacturer'
        except Exception as e:
            return f"Error loading dataset: {str(e)}", None
    return "", None

@app.callback(
    Output('class-dropdown', 'options'),
    Output('class-dropdown', 'value'),
    Input('level-dropdown', 'value')
)
def update_class_dropdown(level):
    if level and level in loaded_datasets:
        dataset = loaded_datasets[level]
        options = [{'label': cls, 'value': cls} for cls in dataset.classes]
        return options, options[0]['value'] if options else None
    return [], None

@app.callback(
    Output('image-output', 'children'),
    Input('level-dropdown', 'value'),
    Input('class-dropdown', 'value')
)
def display_images(level, selected_class):
    if level in loaded_datasets and selected_class:
        dataset = loaded_datasets[level]
        indices = [i for i, (_, cls) in enumerate(dataset.samples) if cls == selected_class]
        selected_indices = random.sample(indices, min(5, len(indices)))

        images = []
        for idx in selected_indices:
            img, cls = dataset[idx]
            if isinstance(img, Image.Image):
                #img = img.resize((224, 224))
                buffer = np.asarray(img)
            else:
                buffer = img.permute(1, 2, 0).numpy()
            fig = px.imshow(buffer)
            fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0))
            images.append(html.Div(dcc.Graph(figure=fig), style={'margin': '10px'}))

        return images
    return []

def find_free_port():
    s = socket.socket()
    s.bind(('', 0))
    return s.getsockname()[1]

port = find_free_port()


if __name__ == '__main__':
    app.run(debug=True, port = port)
# 