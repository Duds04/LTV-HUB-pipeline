all: run test

run:
	@echo "Executando main.py..."
	python3 main.py

test:
	@echo "Executando teste.py..."
	python3 teste.py

clean:
	@echo "Limpando arquivos temporários..."
	clear