# Design

## Introduction

This is a model training tool which takes some images as input and train a fine-tuned generative model.

## Workflow
The API endpoint for this workflow is "https://api.runpod.ai/v2/dream-booth-v1/run". This endpoint will be used in the **Model Training** step.

1. **Data Preparation**: The user prepares a dataset of images, zip them and uploads the zip file to AWS S3 and get a publicly accessible URL.
2. **Model Training**: The user sends a POST request to the `/dream-booth-v1/run` endpoint with the necessary parameters. The parameters include the URL to the zip file, a unique concept name for training, and an optional offset noise parameter for style training.
3. **Model Generation**: The DreamBooth API processes the request and starts training the model. The user can optionally specify a webhook to receive updates about the training process. 
4. **Model Usage**: Once the model is trained, it can be used for generating images based on text inputs.

### Data Preparation

A set of photos is prepared in the `original` directory. For the photos:
- The backgrounds have been removed.
- They contain only the face of the same person.
- They has various sizes and formats (jpg, png)

The `prepare.py` should do the following steps to prepare the dataset:
- Resize all images to the same size 512x512 without twisting or distorting proportions. The resizing be done by adding transparent margins.
- Save the processed images in jpg format in a new directory named `processed`.
- Zip the `processed` directory and upload it to AWS S3.
- Generate a publicly accessible URL for the uploaded zip file.

### Model Trainning

The `train.py` script should perform the following steps to train the model:

- Import the necessary libraries (requests, json, etc.)
- Define the API endpoint URL and the headers (including the RunPod API Token).
- Prepare the payload for the POST request. This includes the URL to the dataset, the unique concept name, and other training parameters.
- Send a POST request to the `/dream-booth-v1/run` endpoint with the prepared payload.
- Handle the response from the server. If the request is successful, print the job ID and status.
- Periodically send a GET request to the `/dream-booth-v1/status/{REQUEST_ID}` endpoint to check the status of the training job.
- Once the job is completed, retrieve the trained model from the provided URL in the response.