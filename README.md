# AI Creator Space

An AI creation web application supporting batch text-to-image and image-to-image generation tasks, with an async background job queue.

## Features
- Multi-image upload via drag-and-drop
- Save and apply your favorite prompt templates
- User authentication with isolated user workspaces and images
- Asynchronous batch queue logic

## Setup

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure users:**
   Copy the `config_example.ini` to `config.ini` and modify the default secure passwords for the users.
   *(Note: `config.ini` is ignored in Git for security)*

3. **Run the application:**
   ```bash
   python run.py
   ```

## Usage
Open `http://127.0.0.1:8000` in your web browser. Log in using the credentials defined in `config.ini` to start generating!