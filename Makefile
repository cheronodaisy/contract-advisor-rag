.PHONY: install build deploy

install:
	pip install -r requirements.txt

build:
	docker build -t myapp:latest .

deploy:
	docker run -d -p 8501:8501 myapp:latest
