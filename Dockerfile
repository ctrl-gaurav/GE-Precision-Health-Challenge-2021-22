FROM python:3.8.3
ADD . /GE-Precision-Health-Challenge-2021-22
WORKDIR /GE-Precision-Health-Challenge-2021-22
RUN pip install -r requirements.txt

CMD [ "python" , "./app.py" ]