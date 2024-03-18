from nvcr.io/nvidia/pytorch:24.02-py3

WORKDIR /usr/src/app

COPY . . 

RUN pip install -r requirements.txt

CMD ["python", "main.py", "train"]
