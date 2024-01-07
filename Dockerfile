FROM python:3-alpine3.17
WORKDIR /app
COPY . /app
RUN pip install -r requirements
EXPOSE 3000
CMD python ./app.py