FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
ENV HOST=0.0.0.0 PORT=8000
# The API key is NOT copied into the image: pass it at runtime (see docker-compose.yml).
# On 0.0.0.0 the "Run evaluation" button is disabled unless ADMIN_VIEW=1 (protects your API quota).
CMD ["python", "server.py"]
