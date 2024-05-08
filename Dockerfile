FROM Python:3.11

WORKDIR .

COPY . .

RUN pip install --uprgrade pip
RUN pip install --no-cache-dir -r requirement.txt

CMD ["python", ".api/_titanic.py"]