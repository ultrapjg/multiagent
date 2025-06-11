FROM python:3.11-slim
WORKDIR /app
COPY requirements/admin-frontend.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY admin-frontend/ /app/admin-frontend/
WORKDIR /app/admin-frontend
EXPOSE 8502
ENV STREAMLIT_SERVER_PORT=8502
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
CMD ["streamlit", "run", "admin_frontend.py", "--server.port=8502", "--server.address=0.0.0.0"]