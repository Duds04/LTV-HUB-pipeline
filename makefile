all: run

run:
	@echo "Se o ambiente não foi configurado, rode o comando 'source ambiente_exec/bin/activate'\n\n"
	@echo "Executando main.py..."
	@python3 main.py

clean:
	@echo "Limpando arquivos temporários..."
	@clear
	@echo "Desative o ambiente virtual com  o comando 'deactivate'"

