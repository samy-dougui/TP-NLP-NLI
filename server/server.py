from flask import Flask, request, make_response
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast


app = Flask(__name__)
device = torch.device("cpu")
model = torch.load("./model.pt").to(device)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
config = {
    "CATEGORY": {
        0: "contradiction",
        1: "neutral",
        2: "entailment"
    }
}


@app.route('/api/prediction', methods=["POST"])
def prediction():
    body = request.get_json()
    try:
        hypothesis = body["hypothesis"]
    except KeyError:
        return make_response({"message": "hypothesis is a required field"}), 400

    try:
        premise = body["premise"]
    except KeyError:
        return make_response({"message": "premise is a required field"}), 400

    inputs = tokenizer(hypothesis, premise, truncation=True,
                       padding='max_length', return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model(inputs["input_ids"].to(device),
                        inputs["attention_mask"].to(device))

        logits = outputs.logits
        softmax = nn.Softmax(dim=1)
        probabilities = softmax(logits)
        index_max = torch.argmax(probabilities, dim=1).item()
        prediction = probabilities[0][index_max]

    return make_response({"results": f'"{hypothesis}" and "{premise}" are: {config["CATEGORY"][index_max]} with a probability of {prediction}',
                          "probabilities": probabilities[0]}), 200


if __name__ == "__main__":
    app.run(port=5000)
