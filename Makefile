.PHONY: build run run-dev stop clean all

IMAGE_NAME = fuzzy-delphi
CONTAINER_NAME = fuzzy-delphi-app
PORT = 8501
PROJECT_DIR = $(shell pwd)

build:
	docker build -t $(IMAGE_NAME) .

# Run with local directory mounted for development
run-dev:
	docker run --name $(CONTAINER_NAME) -p $(PORT):$(PORT) -v $(PROJECT_DIR):/app -d $(IMAGE_NAME)
	@echo "App running in development mode at http://localhost:$(PORT)"
	@echo "Local code changes will be reflected immediately"

stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

clean: stop
	docker rmi $(IMAGE_NAME) || true

dev: stop clean build run-dev
