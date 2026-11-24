# Field of View on the window image and on the visual areas map

In the **Field of View** section of each recording page,
[Build-Notebook-Dataviz-for-All.py](Build-Notebook-Dataviz-for-All.py) builds one figure (`figs/<recording>-FOV.svg`):

| | column 1 | column 2 | column 3 |
|---|---|---|---|
| **top row** | window image + zero (yellow cross) + 2P FOV (square) | window image + visual areas + 2P FOV | – |
| **bottom row** | 2P mean image | 2P max projection | 2P mean image + ROIs |

The 2P images in the bottom row are rotated by 180°, so that they have the same orientation as the window image.

The code is in [../utils_/window_FOV.py](../utils_/window_FOV.py). The visual areas come from [Visual-Areas-Segmentation.py](Visual-Areas-Segmentation.py).

---

## Where the information comes from

| What | Source | Who provides it |
|---|---|---|
| FOV position (µm, relative to the zero) | NWB: `imaging_plane.location` = `coords from 0-ref: (x=..., y=...)` (stage position from the Bruker metadata) | automatic |
| FOV depth | NWB: `imaging_plane.description` = `Depth=...` | automatic |
| FOV size (µm) | 2P image size × `imaging_plane.grid_spacing` (µm/pixel) | automatic |
| Window image | `<dataset>/Processed/intrinsic_img/<day>/<time>/<image>.tif` | **you: DataTable** |
| Scale of the window image (µm/pixel) | `Height-of-Microscope-Camera-Image-in-mm` in the intrinsic `metadata.json` (default 2.7 mm) | automatic |
| Zero position in the window image (pixels) | clicked once with `pick_zero` | **you: DataTable** |
| Visual areas | `segmentation.npy` in the intrinsic session folder | **you: run the segmentation** |

---

## What to do for each new mouse

### 1. Fill in the DataTable

The DataTable is in the dataset folder (the `folder` of Build-Notebook-Dataviz-for-All), next to `Processed/`.
It can be called `DataTable.xlsx` or `DataTable_.xlsx`: the first `DataTable*.xlsx` found is used.

**`Subjects` sheet**, one row per mouse. Add these columns:

| column | example | meaning |
|---|---|---|
| `subject` | `80` | must match the subject ID in the NWB |
| `intrinsic_session` | `2026_11_02/17-28-23` | intrinsic imaging session (`<day>/<time>` inside `Processed/intrinsic_img/`) |
| `window-image` (or `window_image`) | `vasculature` or `217map-sofia.tif` | window image in that session folder (`.tif` is added if missing) |
| `zero_px_x` | `1021` | horizontal position of the zero in the window image (pixels) |
| `zero_px_y` | `615` | vertical position of the zero in the window image (pixels, from the top) |

If a column is empty:
- no `intrinsic_session`: the first intrinsic session whose `metadata.json` has this subject is used, and a warning is printed.
- no `window-image`: the first `vasculature*.tif` of the session is used.
- no `zero_px_x` / `zero_px_y`: the window is shown, but without the cross and the square, and the title says `no zero_px in DataTable`.

**`Recordings` sheet** (optional): if the zero was set again for one recording, add the
`zero_px_x` / `zero_px_y` columns there. For that recording they replace the values of the `Subjects` sheet.
Recordings are matched by the `day` and `time` columns (e.g. `2026_11_03` + `16-29-10`).

### 2. Click the zero (once per mouse, or after setting the zero again)

At the start of each experiment, a 2P image of the zero landmark is saved by the microscope in a folder
whose name contains `zero` (e.g. `TSeries-.../SingleImage-...-zero227/References/...-16bit-Reference.tif`).
`pick_zero` shows the window image (left) and this 2P image of the zero (right, as a reference).
Click on the zero in the window image:

```python
%matplotlib qt          # a window that can be clicked (restart the kernel if it has no effect)
import sys; sys.path += ['../']
from utils_.window_FOV import pick_zero

pick_zero(r'E:\DATA\...\Processed\intrinsic_img\2026_11_02\17-28-23\vasculature.tif',
          zero_image_file=r'E:\DATA\...\Processed\2026_11_03\16-29-10')   # raw session folder of a recording
```

- `zero_image_file` can be the raw session folder: the folder containing `zero` is found inside it automatically.
- The click window waits 120 s (`timeout=0` to wait without limit).
- Copy the printed `zero_px_x = ...    zero_px_y = ...` into the DataTable.

### 3. Run the visual areas segmentation (once per intrinsic session)

In [Visual-Areas-Segmentation.py](Visual-Areas-Segmentation.py):

1. Set `dataFolder` to the **same intrinsic session** as `intrinsic_session` in the DataTable.
2. Adjust `segmentation_params` until the areas look right.
3. Run the **Save the segmentation** cell. It writes into the session folder:
   - `segmentation.npy`: the areas (mask and sign of each area), map size and parameters.
   - `segmentation.png`: the summary figure, to check by eye (written by the Summary Plot cell).
4. Optional: give names to the areas in `AREA_NAMES` (e.g. `{'patch02': 'V1'}`) before saving.

In the window figure, the colors are the same as in the segmentation script: areas with visual field sign
**+1 in red**, **−1 in blue** (transparent, to see the vessels below).

If there is no `segmentation.npy`, the panel says `no segmentation.npy (run Visual-Areas-Segmentation.py)`.

### 4. Build the notebook

Run [Build-Notebook-Dataviz-for-All.py](Build-Notebook-Dataviz-for-All.py) (set `folder` to the dataset folder).
After editing the code, rerun the cell that defines `single_rec`, otherwise the old version is used.

---

## Calibration of the rig (once)

The square is placed correctly only if all of these are right:

| What | Setting | Status |
|---|---|---|
| Square size | from the NWB | reliable |
| Scale of the window image | `Height-of-Microscope-Camera-Image-in-mm` in the intrinsic `metadata.json` (default `DEFAULT_CAMERA_HEIGHT_MM = 2.7`) | **to check** |
| Zero position | `zero_px_x`, `zero_px_y` (your click) | depends on the click |
| Directions of the stage axes in the image | `STAGE_X_SIGN`, `STAGE_Y_SIGN` (+1 or −1) | **to check** |
| Rotation of the stage axes in the image | `STAGE_ANGLE_DEG` | **to check** (0 for now) |

These constants are at the top of [../utils_/window_FOV.py](../utils_/window_FOV.py).

### How to check

- **Blood vessels**: the large surface vessels must be at the same place in the 2P image and in the window image under the square.
  - The vessels are too big or too small: the scale (2.7 mm) is wrong.
  - The vessels are shifted: `zero_px` is wrong.
  - The vessels are mirrored or rotated: `STAGE_X_SIGN` / `STAGE_Y_SIGN` / `STAGE_ANGLE_DEG` are wrong.
  - Check first with the 2P image of the zero (at the surface, so the vessels are clear), then with a shallow recording far from the zero.
- **Directions of the axes**: move the stage by +500 µm in X, then in Y, and take one image each time.
  The image must move in the same direction as the square in the figure.
  The 2P images are displayed rotated by 180°: if the square is mirrored around the cross, try `STAGE_X_SIGN = STAGE_Y_SIGN = -1`.
- **Scale of the window image**: take one picture of a micrometer slide or a ruler with the intrinsic camera,
  and compare the real height of the image with the 2.7 mm of `metadata.json`.
- **Consistency**: recordings of the same mouse at different positions must put their squares on the vessels seen in their 2P images.

---

## Other settings

In [../utils_/window_FOV.py](../utils_/window_FOV.py):

| Setting | Default | Meaning |
|---|---|---|
| `AREA_COLORS` | `{1: 'tab:red', -1: 'tab:blue'}` | color of the areas by visual field sign |
| `alpha` (argument of `show_window_with_areas`) | `0.3` | transparency of the areas |
| `color` (argument of `show_window_with_FOV` / `show_window_with_areas`) | `'green'` | color of the FOV square |
| `DISPLAY_MAX_PX` | `600` | size of the window image in the figure (keeps the `.svg` small; the positions are not affected) |

---

## Messages and what they mean

| Message | What to do |
|---|---|
| `no DataTable*.xlsx in ...` | put the DataTable in the dataset folder |
| `subject "...": no "intrinsic_session" in DataTable, using ...` | fill in `intrinsic_session` for this mouse |
| `no window image found for subject ...` | no intrinsic session for this mouse in `Processed/intrinsic_img/` |
| `no zero_px in DataTable` | click the zero (step 2) and fill in `zero_px_x` / `zero_px_y` |
| `no FOV coordinates in NWB` | old NWB without the stage position: rebuild the NWB with the current physion |
| `no segmentation.npy` | run the segmentation (step 3) |
| `segmentation map (...) and window image (...) have different aspect ratios` | the window image is not from the same camera as the maps: use a `vasculature*.tif` of the segmented session |
