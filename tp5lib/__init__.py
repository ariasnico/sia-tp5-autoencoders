"""tp5lib — lógica reutilizable del TP5 (autoencoders, denoising, VAE).

Se apoya en la librería mlp/ de TP3 (numpy puro). Módulos:
- vae_core   : VAE from-scratch con backprop verificado por gradient check (~5e-08). INTACTO.
- fonts      : carga/exploración de font.h y ruido bit-flip            (se agrega en Fase 1-2).
- autoencoder: build/encode/decode, px_err y loops de entrenamiento    (se agrega en Fase 1-2).
"""
