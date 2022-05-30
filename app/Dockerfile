FROM python:3.8

WORKDIR /app
COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install numpy
RUN pip install sklearn

ENTRYPOINT ["python"]
CMD ["app.py"]
