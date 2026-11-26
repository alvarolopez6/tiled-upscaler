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
  --guidance-scale 3.0 \
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

Los valores actuales por defecto son `guidance_scale=3.0` y un mapeo lineal de
`creativity` a `strength` de `[0,1] → [0.05,0.45]`. Son ajustes educativos
provisionales, no valores óptimos universales.

También se puede usar `UpscalerEngine.upscale()` desde Python. Recibe una imagen
PIL y devuelve un tensor RGB `[1, 3, H, W]` en `[0, 1]`.

## Evaluación visual inicial

Este es un prototipo educativo de difusión, no un sistema de super-resolución
de producción. Comparamos fotos locales de playa (`256×256 → 512×512`) y lago
(`512×512 → 1024×1024`) con baselines Lanczos y VAE-only. Las imágenes locales
de entrada/salida no se incluyen en el repositorio.

Las pruebas controladas usaron semillas fijas, 20 pasos del scheduler,
`tile_size=512`, `overlap=256` y prompts que describían cada escena. Con el mapeo
actual (`[0,1] → strength [0.05,0.45]`), los valores probados corresponden a:

| Creativity | Strength | Pasos efectivos (de 20) | Observación |
|---:|---:|---:|---|
| 0.00 | 0.05 | 1 | Muy conservador; cercano a la reconstrucción VAE. |
| 0.25 | 0.15 | 3 | Conserva la composición general, pero puede quedar suave. |
| 0.50 | 0.25 | 5 | Permite más reinterpretación; no mejora la nitidez de forma consistente. |

Una comparación de CFG con creativity `0.25` encontró que escalas `1–3` suelen
conservar mejor la composición que `7.5`; una guía alta puede producir artefactos
visibles con prompts genéricos. En estas dos fotos, la difusión **no superó a
Lanczos de forma consistente en nitidez**. Es una limitación importante del
modelo/configuración actual; añadir más pasos por sí solo no garantiza resolverla.

## Docker en Linux

La imagen Docker está dirigida a Linux. La configuración base de Compose no
exige GPU, por lo que PyTorch puede usar CPU. En un host Linux con NVIDIA
Container Toolkit, añade el override de NVIDIA para exponer la GPU:

```bash
docker compose build
docker compose run --rm tiled-upscaler --help
```

Ejemplo en CPU:

```bash
docker compose run --rm tiled-upscaler \
  --input /input/photo.jpg \
  --output /output/photo_upscaled.png \
  --steps 20
```

Ejemplo con GPU NVIDIA:

```bash
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml run --rm tiled-upscaler \
  --input /input/photo.jpg \
  --output /output/photo_upscaled.png \
  --steps 20
```

El host necesita un driver NVIDIA compatible y NVIDIA Container Toolkit. La
imagen usa CUDA 13.2.1 para alinearse con las dependencias CUDA 13.2 de la
resolución Linux actual de `uv.lock`. Docker en macOS no puede exponer Apple MPS
a este contenedor Linux; para acelerar en Mac, ejecuta el proyecto de forma
nativa con `uv`.

## Tests

```bash
uv run pytest
```

Los tests cubren coordenadas, máscaras gaussianas, blending NumPy 2D, blending
PyTorch de latentes, preprocesamiento de imágenes y el mapeo de creativity a
strength. La última ejecución reportada pasó 33 casos. Los tests unitarios no
cargan los modelos; la validación de inferencia se hace aparte y requiere
bastante memoria.

## Estructura actual

```text
src/tiled_upscaler/
├── device.py          # selección de dispositivo y precisión
├── preprocessing.py   # imagen PIL → tensores VAE y ControlNet
├── tiling.py          # coordenadas, máscaras y blending
├── engine.py          # modelos y bucle manual de difusión tiled
└── cli.py             # interfaz Typer
Dockerfile             # imagen Linux/CUDA
docker-compose.yml     # servicio Compose compatible con CPU
docker-compose.nvidia.yml # override opcional para GPU NVIDIA
tests/
├── test_tiling.py
├── test_engine.py
└── test_preprocessing.py
assets/input/          # imágenes locales; ignoradas por Git
```

## Estado y limitaciones

- Fases 0–2: implementadas y validadas.
- Fase 3: flujo tiled completo probado con entrada sintética y fotos locales.
  La evaluación visual muestra que la difusión puede quedar más suave que
  Lanczos; la calidad todavía requiere trabajo.
- Fase 4: CLI `--help` e inferencia real probados en MPS y NVIDIA CUDA.
  El profiling es parcial: MPS informa memoria asignada actual, no pico.
- Fase 5: Docker CPU/NVIDIA implementado y una inferencia NVIDIA ejecutada
  técnicamente. Hay que reconstruir y revalidar tras los últimos ajustes de
  CFG/creativity. Docker CPU todavía no se ha probado.
- El proyecto es experimental. El resultado puede variar según el prompt,
  `creativity`, `resemblance` y `steps`.

La inferencia se ejecuta localmente; Hugging Face se utiliza para descargar los
modelos. Revisa las licencias y condiciones de cada modelo antes de usarlo o
redistribuir resultados.

## Licencia

La licencia del proyecto todavía está pendiente de definir.
