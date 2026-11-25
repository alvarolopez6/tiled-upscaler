# Reescalado de imágenes con difusión latente por parches

[English](README.md) | [Español](README.es.md)

Proyecto educativo de ampliación de imágenes con Stable Diffusion 1.5 y
ControlNet Tile. Los componentes del modelo y el bucle de difusión se
implementan explícitamente; no se utiliza un pipeline opaco de alto nivel.

**Estado: experimental.** El motor completo se ha probado en Apple Silicon con
MPS, incluido el denoising por parches. El resultado puede cambiar detalles y no
garantiza conservar cada píxel de la imagen original.

## Qué hace

El motor redimensiona la imagen, la codifica en latentes con el VAE, añade ruido
según `creativity` y lo elimina progresivamente con UNet, ControlNet Tile,
Classifier-Free Guidance y el scheduler Euler Ancestral. Las predicciones de
ruido se calculan por parches y se mezclan antes de actualizar los latentes
globales.

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

- Python 3.12 (indicado en `.python-version`).
- [`uv`](https://docs.astral.sh/uv/).
- Memoria suficiente para cargar Stable Diffusion 1.5 y ControlNet Tile.
- Acceso a Hugging Face Hub para descargar los modelos la primera vez.

El dispositivo se selecciona automáticamente: CUDA, MPS y después CPU. En GPU
se usa `float16`; en CPU, `float32`. El tiling reduce la memoria de activaciones,
pero **los pesos de los modelos siguen ocupando varios GB**. La inferencia en CPU
puede ser muy lenta; una GPU con poca VRAM quizá necesite parches más pequeños.

## Instalación

```bash
uv sync
```

Los pesos se descargan a la caché de Hugging Face la primera vez. Configura
`HF_HOME` si quieres usar otra ubicación. Los pesos y las imágenes de entrada no
se guardan en este repositorio.

## Uso del CLI

Consultar las opciones sin iniciar inferencia:

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

`tile-size` y `overlap` se indican en píxeles de salida y deben ser compatibles
con el factor espacial del VAE (8 para SD 1.5). `steps` es la longitud del
calendario; `creativity` determina cuántos pasos se usan realmente. El CLI
informa del dispositivo, parches por paso, pasos efectivos y tiempo. En CUDA
muestra el pico de memoria asignada; en MPS, la memoria asignada actual, no el
pico.

También se puede usar `UpscalerEngine.upscale()` desde Python. Recibe una imagen
PIL y devuelve un tensor RGB `[1, 3, H, W]` en `[0, 1]`.

## Tests

```bash
uv run pytest
```

Los tests cubren coordenadas, máscaras gaussianas, blending NumPy 2D, blending
PyTorch de latentes y preprocesamiento de imágenes. La última ejecución
reportada pasó 28 casos; se ejecutó antes de los cambios recientes del CLI. Los
tests unitarios no cargan los modelos; la validación de inferencia se hace aparte
y requiere bastante memoria.

## Estructura actual

```text
src/tiled_upscaler/
├── device.py          # selección de dispositivo y precisión
├── preprocessing.py   # imagen PIL → tensores VAE y ControlNet
├── tiling.py          # coordenadas, máscaras y blending
├── engine.py          # modelos y bucle manual de difusión tiled
└── cli.py             # interfaz Typer
tests/
├── test_tiling.py
└── test_preprocessing.py
assets/input/          # imágenes locales; ignoradas por Git
```

## Estado y limitaciones

- Fases 0–2: implementadas y validadas.
- Fase 3: flujo tiled completo implementado y probado con una entrada sintética
  y una fotografía local. Falta seguir evaluando calidad, memoria y distintos
  tamaños de entrada.
- Fase 4: CLI implementado y `--help` comprobado. Falta validar una inferencia
  lanzada desde el CLI y completar el profiling.
- Fase 5: Docker pendiente.
- El proyecto es experimental. El resultado puede variar según el prompt,
  `creativity`, `resemblance` y `steps`.

La inferencia se ejecuta localmente; Hugging Face se utiliza para descargar los
modelos. Revisa las licencias y condiciones de cada modelo antes de usarlo o
redistribuir resultados.

## Licencia

La licencia del proyecto todavía está pendiente de definir.
