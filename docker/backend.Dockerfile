FROM python:3.10-slim
WORKDIR /app
COPY requirements/backend.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY mcp-config ./mcp-config
EXPOSE 8000
CMD ["python", "backend/main.py"]
