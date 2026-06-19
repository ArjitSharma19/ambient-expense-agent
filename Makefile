# Makefile for Ambient Expense Agent

.PHONY: install playground run

install:
	uv sync

playground:
	uv run agents-cli playground

run:
	uv run python app/fast_api_app.py
