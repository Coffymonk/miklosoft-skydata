# miklosoft-skydata

Cloud and rain maps for the **Radar** tab of the Sky planner at [miklosoft.se](https://miklosoft.se).

Every 15 minutes a scheduled GitHub job collects open weather data, turns it into one set of
maps and publishes them here on GitHub Pages. The website fetches the maps from here, so the
people who use the planner never contact these services themselves.

## What it makes

For Europe, on one map grid (Web Mercator, about 2–3 km per pixel):

- **The last two hours as measured**, every 15 minutes:
  - clouds from the Meteosat satellites
  - rain from the European weather radars
- **The next two hours as a nowcast.** The program measures how the clouds and showers moved
  over the last half hour and carries them on.
- **The next 18 hours from forecasts:** DWD's ICON-EU model and the ECMWF ensemble (50
  forecasts at once).

Each forecast pixel holds a **chance of cloud**, not just "cloud" or "clear". The chance comes
from two things:

- how much the cloud could shift in that time
- how many of the ensemble forecasts bring cloud

## How often it is right

Every run keeps its forecasts. Later runs compare them with what the satellite actually saw
over land, and keep a 30-day tally of how often each lead time (15 minutes … 12 hours) was right.

The tally is published, so the website can show the real hit rate instead of a promise. The same
tally corrects the percentages: if "70 %" turns out to mean cloud 8 times in 10, the program
says 80 % from then on.

## Sources, licences and credits

| Source | Used for | Licence | Credit |
|---|---|---|---|
| [EUMETSAT](https://www.eumetsat.int) via EUMETView | Meteosat Cloud Mask (MSG, 0°) and MTG infrared 10.5 µm images | EUMETSAT data policy: Core data, free and unrestricted | *Contains modified EUMETSAT Meteosat data (year)* |
| [EUMETNET OPERA](https://www.eumetnet.eu/activities/observations-programme/current-activities/opera/) | European radar composite (maximum reflectivity) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | *Radar data: EUMETNET OPERA, CC BY 4.0, modified* |
| [Deutscher Wetterdienst](https://opendata.dwd.de) | ICON-EU total cloud cover, precipitation, ground temperature | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | *Source: Deutscher Wetterdienst (DWD), ICON-EU, modified* |
| [ECMWF](https://www.ecmwf.int) open data | Ensemble total cloud cover | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | *Copyright ECMWF, CC BY 4.0, modified* |

All data here has been modified: resampled, combined, moved forward in time and turned into
chances. ECMWF does not accept any liability for errors or omissions in the data, its availability,
or any loss or damage arising from its use; nor do the other providers or this project.

**Software:**
- [NumPy](https://numpy.org) and [SciPy](https://scipy.org) (BSD)
- [OpenCV](https://opencv.org) (Apache 2.0)
- [h5py](https://www.h5py.org) (BSD)
- [pyproj](https://pyproj4.github.io/pyproj/) (MIT)
- [ecCodes](https://github.com/ecmwf/eccodes) (Apache 2.0)
- [Pillow](https://python-pillow.org) (MIT-CMU)
- [Requests](https://requests.readthedocs.io) (Apache 2.0)

## Being a good neighbour

- Each run asks for a picture or file only once and keeps what it already has.
- Only the cloud cover is taken from the ensemble files, by asking for byte ranges.
- Every request names this project and a contact address.
- One failing source never stops the others.

## The published files

- `manifest.json`: the frame times, data sources, motion field, accuracy tally and credits.
- `tiles/{set}/{x}_{y}.bin.gz`: one 128 × 128 square of the map at zoom 5, all frames in a row.
  - One byte per pixel: high 4 bits are the chance of cloud in tenths (15 = no data), low 4 bits the rain class (15 = unknown).
- `accuracy.json`: the full tally, including how reliable each percentage was.

## Running it yourself

```
pip install -r requirements.txt
python -m skydata.run --state state --out out
```

## Contact

stargazer@miklosoft.se

The code is MIT licensed (see `LICENSE`). The data keeps its own licences, listed above. 
