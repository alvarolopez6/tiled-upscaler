# Tiled Latent Diffusion Upscaler

> An educational image-upscaling project built with Stable Diffusion 1.5 and
> ControlNet Tile. The diffusion components are loaded and called explicitly;
> this project does not use an opaque high-level image-generation pipeline.

**Estado: experimental.** The end-to-end engine has been exercised on an Apple
Silicon Mac with MPS, including tiled denoising. Results can change image
details and are not guaranteed to preserve every pixel of the input.

## Qué hace

El motor redimensiona una imagen, la codifica en latentes con el VAE, añade
ruido según `creativity`, y ejecuta denoising con UNet, ControlNet Tile,
Classifier-Free Guidance y Euler Ancestral. Las predicciones se calculan por
parches y se mezclan antes de actualizar los latentes globales.

```text
imagen + prompt
      ↓
VAE → latentes con ruido
      ↓
ControlNet Tile + UNet + scheduler, por parches
      ↓
VAE → imagen ampliada
```

## Requisitos

- Python 3.12 (versión fijada en `.python-version`).
- [`uv`](https://docs.astral.sh/uv/).
- Memoria suficiente para cargar Stable Diffusion 1.5 y ControlNet Tile.
- Hugging Face Hub para descargar los modelos la primera vez.

El dispositivo se selecciona automáticamente: CUDA, MPS y finalmente CPU. En
GPU se usa `float16`; en CPU, `float32`. El procesamiento tiled reduce memoria
de activaciones, pero **los pesos de los modelos siguen ocupando varios GB**.
CPU puede ser muy lento y una GPU con poca VRAM puede requerir tiles menores.

## Instalación

```bash
uv sync
```

Los pesos se descargan a la caché de Hugging Face en el primer uso. Se puede
usar `HF_HOME` para elegir otra ubicación. Los modelos y las imágenes de entrada
no forman parte del repositorio.

## Uso del CLI

Ver las opciones sin cargar los modelos:

```bash
uv run tiled-upscaler --help
```

Ejemplo:

```bash
uv run tiled-upscaler \
  --input photo.jpg \
  --output "$HOME/Desktop/photo_upscaled.png" \
  --scale 2 \
  --creativity 0.25 \
  --resemblance 0.85 \
  --steps 20 \
  --tile-size 512 \
  --overlap 256 \
  --prompt "high resolution landscape photograph, sharp focus, detailed textures"
```

`tile-size` y `overlap` están en píxeles de salida y deben ser compatibles con
el factor espacial del VAE (8 para SD 1.5). `steps` es el número de pasos que
prepara el scheduler; `creativity` selecciona cuántos se usan efectivamente.
El CLI imprime dispositivo, número de parches por paso, pasos efectivos y
tiempo. En CUDA muestra memoria pico asignada; en MPS muestra memoria asignada
actual, no el pico.

También se puede usar el motor desde Python con `UpscalerEngine.upscale()`. El
método recibe una imagen PIL y devuelve un tensor RGB `[1, 3, H, W]` en `[0, 1]`.

## Tests

```bash
uv run pytest
```

Los tests cubren coordenadas, máscaras gaussianas, blending NumPy 2D, blending
de latentes PyTorch y preprocesamiento de imágenes. La última ejecución
reportada pasó 28 casos. Los tests unitarios no cargan los modelos; las
validaciones de inferencia se hacen por separado y requieren bastante memoria.

## Estructura actual

```text
src/tiled_upscaler/
├── device.py          # selección de dispositivo y precisión
├── preprocessing.py   # imagen PIL → tensores para VAE y ControlNet
├── tiling.py          # coordenadas, máscaras y blending
├── engine.py          # modelos y bucle manual de difusión tiled
└── cli.py             # interfaz Typer
tests/
├── test_tiling.py
└── test_preprocessing.py
assets/input/          # fotos locales; ignoradas por Git
```

## Estado y limitaciones

- Fases 0–2: implementadas y validadas.
- Fase 3: el flujo tiled completo está implementado y se ha probado con una
  imagen sintética y con una fotografía local; falta seguir evaluando calidad,
  memoria y distintos tamaños de entrada.
- Fase 4: CLI implementado; `--help` comprobado. Falta validar una inferencia
  ejecutada a través del CLI y completar el profiling.
- Fase 5: Docker todavía pendiente.
- El repositorio es experimental; la imagen puede cambiar y `creativity`,
  `resemblance`, `steps` y el prompt requieren ajuste según el caso.

La inferencia se ejecuta localmente; el acceso a Hugging Face se usa para
descargar los modelos. Comprueba las licencias y condiciones de uso de cada
modelo antes de utilizarlo o redistribuir sus resultados.

## Licencia

La licencia del proyecto todavía está pendiente de definir.
