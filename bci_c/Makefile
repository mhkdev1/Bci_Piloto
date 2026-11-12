# Makefile -- projeto C do BCI/ortese.
#
# Zero dependencias externas: C99 + libm. Nada de torch, nada de MNE, nada de
# BrainFlow. O mesmo Makefile compila no Windows (mingw-w64), no Linux e no
# Raspberry Pi -- e e de proposito que o rede.c seja o mesmo arquivo nos tres.
#
#   make                 compila tudo em artefatos/
#   make tempo           mede ms por janela do forward
#   make limpar
#
# No Pi:  make CFLAGS="-std=c99 -O2 -Wall -mcpu=arm1176jzf-s -mfpu=vfp"
# (ARMv6 nao tem NEON; -mfpu=neon NAO compila no Zero v1.3)

CC      := gcc
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -Wpedantic
LDLIBS  ?= -lm

OBJ := construcao
ART := artefatos

# O gcc do mingw acrescenta .exe sozinho. Sem detectar isso, o alvo nunca existe
# para o make e tudo e reconstruido a cada chamada. O $(OS) nao serve: dentro do
# shell do MSYS2 ele nao vem como Windows_NT.
UNAME := $(shell uname -s 2>/dev/null)
ifneq (,$(findstring MINGW,$(UNAME))$(findstring MSYS,$(UNAME))$(findstring CYGWIN,$(UNAME)))
EXE := .exe
else ifeq ($(OS),Windows_NT)
EXE := .exe
else
EXE :=
endif

BANCADA := $(ART)/bancada$(EXE)

.PHONY: todos limpar tempo paridade
todos: $(BANCADA)

$(OBJ)/rede.o: src/rede.c src/rede.h
	mkdir -p $(OBJ)
	$(CC) $(CFLAGS) -Isrc -c $< -o $@

$(OBJ)/bancada.o: alvos/bancada.c src/rede.h
	mkdir -p $(OBJ)
	$(CC) $(CFLAGS) -Isrc -c $< -o $@

$(BANCADA): $(OBJ)/bancada.o $(OBJ)/rede.o
	mkdir -p $(ART)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

tempo: $(BANCADA)
	$(BANCADA) tempo $(ART)/modelo.bin 100

paridade: $(BANCADA)
	python teste/comparar.py

limpar:
	rm -rf $(OBJ) $(ART)/bancada$(EXE) $(ART)/paridade_*.bin
