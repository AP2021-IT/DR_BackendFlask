FROM python:3
WORKDIR /usr/src/app
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .
CMD ["python3","app.py"]