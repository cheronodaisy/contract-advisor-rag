.PHONY: install build deploy

install:
	pip install -r requirements.txt

build:
	docker build -t myapp:latest .

deploy:
	docker run -d -p 80:80 myapp:latest
