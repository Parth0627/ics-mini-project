PYTHON := .venv/bin/python

.PHONY: setup test demo doctor
setup:
	sh scripts/bootstrap.sh
test:
	$(PYTHON) -m pytest -q
demo:
	$(PYTHON) -m dns_exfiltration demo
doctor:
	$(PYTHON) -m dns_exfiltration doctor
