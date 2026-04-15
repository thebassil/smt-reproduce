.PHONY: install inspect-all test docker

install:
	pip install -e ".[train]"

inspect-all:
	smt status
	smt reproduce systems --inspect
	smt reproduce subsystem --inspect
	smt reproduce cross-system --inspect
	smt reproduce ground-truth --inspect

test:
	smt status
	smt reproduce systems --inspect --system machsmt
	smt reproduce cross-system --inspect --axis model

docker:
	docker build -t smt-reproduce:latest .
	docker run --rm -v ./data:/app/data smt-reproduce:latest status
