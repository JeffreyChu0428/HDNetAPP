from fastapi import FastAPI, WebSocket
import torch
import torch.nn as nn
from torchhd.embeddings import Projection
from torchhd.models import Centroid
from model.model_def import HDNet, PTBXL_Dataset, Patient_Dataset, ECGNet, Embedding, SiameseNet

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

abnormal_train_database = PTBXL_Dataset("model/ptbxl_abnormal_data.pt", split='val')
normal_train_database = PTBXL_Dataset("model/ptbxl_normal_data.pt")

model = HDNet(normal_train_database, abnormal_train_database, "model/NormalClassifier.pth", "model/ECGNet_weightedLoss.pth")
model.load_state_dict(torch.load("model/hdnet.pth"))
model.to(device)

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        data = await websocket.receive_json()
        action = data.get("action")

        if action == "inference":
            input_data = data.get("data")
            model.eval()

            test_data = Patient_Dataset(input_data)
            pred_labels, normal_similarities = model.test(test_data)
            print('Input Pridicted!')

            await websocket.send_json({
                "type": "inference_result",
                "result": pred_labels
            })

        elif action == "training":
            train_data = data.get("data")
            label = data.get("label")

            print("Data Collected!")
            train_database = Patient_Dataset(train_data, label)
            model.custom_train(train_database)

            await websocket.send_json({
                "type": "training_done"
            })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)