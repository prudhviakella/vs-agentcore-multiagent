FROM public.ecr.aws/lambda/python:3.11
COPY requirements.txt .
RUN pip install -r requirements.txt --target /var/task
COPY handler.py /var/task/
CMD ["handler.handler"]
