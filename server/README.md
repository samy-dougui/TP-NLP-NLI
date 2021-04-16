## Server

This is an API example to test our trained model using http request.

## How to

To build the container, run the following command: `docker build -t server .`

To run the container (in interactive mode): `docker run -it --rm -p 5000:5000 server`

To test a model you simply need to put the model into the server folder and update the path in the server.py

The body of the request must be of the following shape:

```json
{
  "hypothesis": "sentence1",
  "premise": "sentence2"
}
```

If instead of using Postman or any other tool, you want to use cURL:

```bash
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"hypothesis":"I like NLP.","premise":"This is a difficult task."}' \
  http://localhost:5000/api/prediction
```
