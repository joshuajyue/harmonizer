FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir fastapi "uvicorn[standard]"
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000"]