import marimo

__generated_with = "0.22.0"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import geopandas as gpd
    import requests
    from tqdm import tqdm

    from lonboard import Map, PolygonLayer


    return Map, Path, PolygonLayer, gpd, requests, tqdm


@app.cell
def _():
    url = "https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/CGAZ/geoBoundariesCGAZ_ADM1.geojson"
    filepath = "shird_urban/geoBoundariesCGAZ_ADM1.geojson"
    return filepath, url


@app.cell
def _(Path, filepath, requests, tqdm, url):
    if not Path(filepath).exists():
        # Make a GET request to download the file
        # Streaming, so we can iterate over the response and create a progress bar for this large file
        response = requests.get(url, stream=True)

        # Sizes in bytes.
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024

        with (
            tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar,
            open(filepath, "wb") as file,
        ):
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

        if total_size not in (0, progress_bar.n):
            raise RuntimeError("Could not download file")
    return


@app.cell
def _(filepath, gpd):
    gdf = gpd.read_file(filepath, engine="pyogrio", use_arrow=True)

    # Keep columns necessary for our visualization
    cols = ["shapeName", "shapeID", "geometry"]
    gdf = gdf[cols]
    return (gdf,)


@app.cell
def _(gdf):
    eastern_hemisphere = gdf.cx[0:180, -90:90]
    return (eastern_hemisphere,)


@app.cell
def _(PolygonLayer, eastern_hemisphere):
    layer = PolygonLayer.from_geopandas(
        eastern_hemisphere,
        get_line_width=20,  # width in default units (meters)
        line_width_min_pixels=0.2,  # minimum width when zoomed out
        get_fill_color=[204, 251, 254],  # light blue
        get_line_color=[37, 36, 34],  # dark border color
    )
    return (layer,)


@app.cell
def _(Map, layer):
    m = Map(layer)
    m
    return


if __name__ == "__main__":
    app.run()
