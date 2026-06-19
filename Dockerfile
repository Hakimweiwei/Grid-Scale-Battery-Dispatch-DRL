FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --upgrade pip && pip install -e .

# Copy source code
COPY . .

# Expose Streamlit port
EXPOSE 8501

CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
