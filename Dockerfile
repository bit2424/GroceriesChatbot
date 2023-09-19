# Use an official TensorFlow GPU image as the base image
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory inside the container
WORKDIR /app

# Copy your LLM fine-tuning project files into the container
COPY . .

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt
