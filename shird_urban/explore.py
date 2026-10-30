import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import geopandas as gpd
    import marimo as mo
    import polars as pl
    from lonboard import Map, PolygonLayer, viz  # noqa: F401

    return Map, PolygonLayer, gpd, mo, pl


@app.cell
def _(gpd):
    shp_path = "~/Downloads/shrug-shrid-poly-shp/shrid2_open.shp"

    shp = gpd.read_file(shp_path)
    shp = shp.set_index("shrid2")
    shp
    return (shp,)


@app.cell
def _(Map, PolygonLayer):
    def plot_shrid(gdf, df):
        filt_gdf = gdf.join(df.select("shrid2").to_pandas().set_index("shrid2"), how="inner")
        print(f"{filt_gdf.shape[0]} records after filtering")
        layer = PolygonLayer.from_geopandas(
            filt_gdf,
            get_fill_color=[255, 0, 0],
            get_line_color=[0, 0, 0],
        )
        m = Map(layer)

        return m

    return (plot_shrid,)


@app.cell
def _(pl):
    shrid_frag_con08_key = pl.read_csv("~/Downloads/shrug-con-keys-csv/shrid_frag_con08_key.csv")
    shrid_frag_con08_key
    return (shrid_frag_con08_key,)


@app.cell
def _(pl):
    ac08_name_key = pl.read_csv("~/Downloads/shrug-con-keys-csv/ac08_name_key.csv")
    ac08_name_key
    return


@app.cell
def _(pl, shrid_frag_con08_key):
    ## are there are shrids in multiple acs?
    shrid_frag_con08_key.group_by("shrid2").agg(pl.col("ac08_id").n_unique()).filter(pl.col("ac08_id") > 1)
    return


@app.cell
def _(pl, shrid_frag_con08_key):
    tmp = shrid_frag_con08_key.filter(pl.col("ac08_id").str.starts_with("2008-20-"))
    tmp.shape
    return (tmp,)


@app.cell
def _(mo, plot_shrid, shp, tmp):
    chart = plot_shrid(shp, tmp.head(20))
    mo.ui.anywidget(chart)
    return


if __name__ == "__main__":
    app.run()
