import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import geopandas as gpd
    import polars as pl
    import polars_st as st
    import leafmap.foliumap as lmap

    return gpd, lmap, pl, st


@app.cell
def _(pl):
    triv_cand = pl.read_csv("data/tabular/shrug/shrug-triv-cand-csv/trivedi_candidates_clean.csv")
    return (triv_cand,)


@app.cell
def _(gpd):
    df = gpd.read_file("data/geo/assembly/assembly_boundaries.geojson")
    df = df.set_crs(epsg=3857, allow_override=True)
    df = df.to_crs(epsg=4326)
    df.loc[:,"geometry"] = df.geometry.make_valid()
    return (df,)


@app.cell
def _(df):
    ## Correction -- 
    df.loc[df.uid == "3400000155", "uid"] = "3300000155"
    return


@app.cell
def _(df, lmap):
    m = lmap.Map()
    m.add_gdf(df[df.state_ut == "Sikkim"])
    m
    return


@app.cell
def _(df, st):
    pl_gdf = st.from_geopandas(df).with_columns(
        st.geom("geometry").st.set_srid(4326).alias("geometry")
    )
    return (pl_gdf,)


@app.cell
def _(pl, triv_cand):
    _ac_num = pl.col("ac08_id").str.slice(8, 3).cast(pl.Int64)
    _is_tlg = pl.col("sub_region") == "telangana"
    _is_ap_new = (pl.col("pc01_state_id") == 28) & ~_is_tlg
    _state_prefix = (
        pl.when(_is_tlg).then(pl.lit("36"))
        .otherwise(pl.col("ac08_id").str.slice(5, 2))
    )
    _remapped_num = (
        pl.when(_is_ap_new).then(_ac_num - 119)
        .otherwise(_ac_num)
    )
    bjp_cands = (
        triv_cand.filter(pl.col("ac08_id").is_not_null())
        .filter(pl.col("party") == "BJP")
        .with_columns(
            (
                _state_prefix
                + pl.lit("00000")
                + _remapped_num.cast(pl.Utf8).str.zfill(3)
            ).alias("geo_uid")
        )
    )
    bjp_cands
    return (bjp_cands,)


@app.cell
def _(bjp_cands, pl_gdf):
    bjp_geo = pl_gdf.join(bjp_cands, left_on="uid", right_on="geo_uid", how="inner")
    return (bjp_geo,)


@app.cell
def _(bjp_geo, pl):
    bjp_geo_avg_vsp = bjp_geo.group_by("uid").agg(
        pl.col("geometry").first(),
        pl.col("state_ut").first(),
        pl.col("cons_name").first(),
        pl.col("cons_code").first(),
        pl.col("ac08_id").first(),
        pl.col("tr_ac_name").first(),
        pl.col("ac08_name").first(),
        (pl.col("vote_share_percentage") * pl.col("valid_votes")).sum() /
        pl.col("valid_votes").sum(),

    )
    return (bjp_geo_avg_vsp,)


@app.cell
def _(bjp_geo_avg_vsp, pl):
    bjp_geo_avg_vsp.with_columns(pl.col("geometry").st.simplify(tolerance=0.01)).st.plot().encode(color="vote_share_percentage").project("mercator").properties(width="container")
    return


@app.cell
def _(gpd):
    states_gdf = gpd.read_file(
        "data/geo/humdata/geoBoundaries-IND-ADM1-all/geoBoundaries-IND-ADM1_simplified.geojson"
    )
    return (states_gdf,)


@app.cell
def _(bjp_geo_avg_vsp, df, states_gdf):
    import unicodedata
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import matplotlib.cm as _cm
    import matplotlib.colors as _mcolors

    _bjp_pd = bjp_geo_avg_vsp.select(["uid", "vote_share_percentage"]).to_pandas()
    _ac_bjp = df.merge(_bjp_pd, on="uid", how="left").rename(
        columns={"vote_share_percentage": "bjp_vote_share"}
    )
    _states = states_gdf.copy()
    _states["shapeNameAscii"] = _states["shapeName"].map(
        lambda s: unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    )

    plt.rcParams["font.family"] = "Avenir Next"
    BG, LAND, GREY_LINE, DARK_LINE = "#fbf9f4", "#ececea", "#b3b3b3", "#1f1f1f"
    INK, MUTED, LEADER = "#181818", "#5b5b5b", "#7a7a7a"

    _xmin, _ymin, _xmax, _ymax = 65.5, 6.0, 99.5, 37.5
    _W, _H = _xmax - _xmin, _ymax - _ymin
    _map_h = 14 * (_H / _W)
    _top_chrome, _bot_chrome = 1.7, 2.0
    _fig_h = _top_chrome + _map_h + _bot_chrome

    _fig = plt.figure(figsize=(14, _fig_h), dpi=150, facecolor=BG)
    _ax = _fig.add_axes([0.04, _bot_chrome / _fig_h, 0.92, _map_h / _fig_h])
    _ax.set_facecolor(BG)

    _states.plot(ax=_ax, facecolor=LAND, edgecolor=GREY_LINE, linewidth=0.4)
    _ac_bjp[_ac_bjp.bjp_vote_share.notna()].plot(
        ax=_ax, column="bjp_vote_share", cmap="Oranges", vmin=0, vmax=100,
        edgecolor="white", linewidth=0.06,
    )
    _states.plot(ax=_ax, facecolor="none", edgecolor=DARK_LINE, linewidth=0.7)
    _ax.set_xlim(_xmin, _xmax)
    _ax.set_ylim(_ymin, _ymax)
    _ax.set_aspect("equal")
    _ax.set_axis_off()

    _fig.text(0.04, 1 - 0.55/_fig_h, "BJP vote share by Assembly Constituency",
              fontsize=32, fontweight="bold", color=INK, ha="left", va="center",
              path_effects=[pe.withStroke(linewidth=3, foreground=BG)])
    _fig.text(0.04, 1 - 1.10/_fig_h,
              "Weighted-average BJP candidate vote share, all elections in the Trivedi candidates dataset",
              fontsize=14, color=MUTED, ha="left", va="center")

    _cax = _fig.add_axes([0.30, 1.45/_fig_h, 0.40, 0.014])
    _sm = _cm.ScalarMappable(cmap="Oranges", norm=_mcolors.Normalize(vmin=0, vmax=100))
    _cb = _fig.colorbar(_sm, cax=_cax, orientation="horizontal")
    _cb.outline.set_visible(False)
    _cb.ax.tick_params(labelsize=11, color=MUTED, length=3, pad=3)
    _cb.set_ticks([0, 25, 50, 75, 100])
    for _lbl in _cb.ax.get_xticklabels():
        _lbl.set_color(INK)
    _cax.text(0.5, 2.1, "BJP vote share (%)", fontsize=12, color=INK, fontweight="bold",
              transform=_cax.transAxes, ha="center", va="bottom")

    _ax.text(83.0, 14.5, "READING THE MAP",
             fontsize=12.5, color=INK, fontweight="bold", ha="left", va="top",
             path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)], zorder=12)
    _commentary = (
        "The BJP's heartland is the Hindi-\n"
        "belt north and west — Gujarat,\n"
        "Madhya Pradesh, Rajasthan, Uttar\n"
        "Pradesh — where deep oranges\n"
        "dominate. Punjab is a notable\n"
        "northern exception. In the south,\n"
        "the BJP is largely absent except\n"
        "in Karnataka, where it competes\n"
        "seriously for state power.\n"
        "Maharashtra shows a patchier\n"
        "picture, reflecting the BJP's\n"
        "reliance on coalition partners\n"
        "(Shiv Sena factions)."
    )
    _ax.text(83.0, 13.7, _commentary,
             fontsize=10.5, color=INK, ha="left", va="top", linespacing=1.55,
             path_effects=[pe.withStroke(linewidth=2, foreground=BG)], zorder=12)

    _state_focus = [
        ("Punjab",       (68.5, 34.0)),
        ("Maharashtra",  (66.5, 21.0)),
        ("Karnataka",    (66.5, 15.5)),
    ]
    _states_idx = _states.set_index("shapeNameAscii")
    for _name, (_lx, _ly) in _state_focus:
        _row = _states_idx.loc[_name]
        _cx, _cy = _row.geometry.representative_point().coords[0]
        _ax.annotate(
            _name, xy=(_cx, _cy), xytext=(_lx, _ly),
            fontsize=12, color=INK, fontweight="bold", ha="left", va="center",
            path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)],
            arrowprops=dict(arrowstyle="-", color=LEADER, lw=0.7, shrinkA=2, shrinkB=5),
            zorder=11,
        )

    import textwrap as _textwrap
    _sources_line = (
        "Sources  Trivedi Centre for Political Data (Ashoka University) candidates dataset · "
        "Assembly constituency boundaries from SHRUG / Election Commission of India · "
        "State boundaries from geoBoundaries (HDX, 2023)."
    )
    _methodology_line = (
        "Methodology  Per-AC weighted average of BJP candidate vote share across every election present in the dataset, "
        "weighted by valid votes cast. ACs without a BJP candidate match in the dataset are shown in the basemap colour."
    )
    _caption = (
        _textwrap.fill(_sources_line, width=155) + "\n"
        + _textwrap.fill(_methodology_line, width=155)
    )
    _fig.text(0.04, 0.02, _caption, fontsize=9, color=MUTED, ha="left", va="bottom", linespacing=1.7)

    _out = "urban/bjp_vote_share_map.png"
    _fig.savefig(_out, dpi=200, facecolor=BG)
    plt.close(_fig)
    _out
    return


@app.cell
def _(pl, st):
    worldpop_1km_2020 = pl.read_csv("data/geo/worldpop/ind_pd_2020_1km_ASCII_XYZ.csv")
    worldpop_1km_2020 = worldpop_1km_2020.select(
        geometry=st.point(pl.concat_arr("X", "Y")).st.set_srid(4326),
        pop_int=pl.col("Z"),
        px=pl.col("X"),
        py=pl.col("Y"),
        xb=pl.col("X").floor().cast(pl.Int32),
        yb=pl.col("Y").floor().cast(pl.Int32),
    )
    return (worldpop_1km_2020,)


@app.cell
def _(pl, worldpop_1km_2020):
    worldpop_1km_2020[400_000:420_000].with_columns(pl.col("geometry").st.simplify(tolerance=0.01)).st.plot().encode(color="pop_int").project("mercator").properties(width="container")
    return


@app.cell
def _(worldpop_1km_2020):
    worldpop_1km_2020
    return


@app.cell
def _(bjp_geo_avg_vsp, pl, st, worldpop_1km_2020):
    import os as _os

    _cache_path = "urban/pop_by_ac.parquet"
    if _os.path.exists(_cache_path):
        pop_by_ac = pl.read_parquet(_cache_path)
    else:
        _poly = bjp_geo_avg_vsp.select(["uid", "geometry"]).with_columns(
            _bounds=st.geom("geometry").st.bounds(),
        ).with_columns(
            minx=pl.col("_bounds").arr.get(0),
            miny=pl.col("_bounds").arr.get(1),
            maxx=pl.col("_bounds").arr.get(2),
            maxy=pl.col("_bounds").arr.get(3),
        ).drop("_bounds").with_columns(
            xb=pl.int_ranges(
                pl.col("minx").floor().cast(pl.Int32),
                pl.col("maxx").floor().cast(pl.Int32) + 1,
            ),
            yb=pl.int_ranges(
                pl.col("miny").floor().cast(pl.Int32),
                pl.col("maxy").floor().cast(pl.Int32) + 1,
            ),
        ).explode("xb").explode("yb")

        _merged = _poly.lazy().join(
            worldpop_1km_2020.lazy().select(["geometry", "pop_int", "px", "py", "xb", "yb"]),
            on=["xb", "yb"],
            how="inner",
            suffix="_pt",
        ).filter(
            (pl.col("px") >= pl.col("minx"))
            & (pl.col("px") <= pl.col("maxx"))
            & (pl.col("py") >= pl.col("miny"))
            & (pl.col("py") <= pl.col("maxy"))
        ).filter(
            st.geom("geometry").st.contains(st.geom("geometry_pt"))
        )

        pop_by_ac = (
            _merged.group_by("uid")
            .agg(
                pop_total=pl.col("pop_int").sum(),
                n_cells=pl.len(),
            )
            .collect(engine="streaming")
        )
        pop_by_ac.write_parquet(_cache_path)
    pop_by_ac
    return (pop_by_ac,)


@app.cell
def _(bjp_geo_avg_vsp):
    bjp_geo_avg_vsp[0:1]
    return


@app.cell
def _(bjp_geo_avg_vsp, pl, pop_by_ac):
    _bjp = bjp_geo_avg_vsp.select(["uid", "state_ut", "cons_name", "vote_share_percentage"])
    scatter_df = (
        pop_by_ac.join(_bjp, on="uid", how="inner")
        .with_columns(pop_density=pl.col("pop_total") / pl.col("n_cells"))
        .with_columns(
            vote_share_dev=(
                pl.col("vote_share_percentage")
                - pl.col("vote_share_percentage").mean().over("state_ut")
            )
        )
    )
    scatter_df
    return (scatter_df,)


@app.cell
def _(scatter_df):
    import matplotlib.pyplot as _plt
    from matplotlib.lines import Line2D as _Line2D
    import numpy as _np

    _excluded = ("Punjab", "Kerala", "Tamil Nadu", "Andhra Pradesh", "Telangana")
    _excl_short = "Punjab, Kerala, TN, AP, Telangana"
    _bg, _orange, _teal, _all_trend = "#fbf9f4", "#d95f0e", "#1b9e9e", "#1f1f1f"

    _d = scatter_df.to_pandas().dropna(subset=["pop_density", "vote_share_dev"])
    _d = _d[_d["pop_density"] > 0]
    _is_excl = _d["state_ut"].isin(_excluded)
    _log_d = _np.log10(_d["pop_density"].to_numpy())
    _y = _d["vote_share_dev"].to_numpy()

    _fig, _ax = _plt.subplots(figsize=(8, 6), dpi=130, facecolor=_bg)
    _ax.set_facecolor(_bg)
    _ax.scatter(_d["pop_density"], _y,
                s=30, c=_is_excl.map({True: _teal, False: _orange}),
                alpha=0.3, edgecolor=_bg, linewidth=0.5)
    _ax.set_xscale("log")
    _ax.axhline(0, color="#444", linewidth=0.8, linestyle="--", alpha=0.6)

    def _fit_line(_x, _yv, _color, _ls):
        if len(_x) < 2:
            return
        _m, _b = _np.polyfit(_x, _yv, 1)
        _xg = _np.linspace(_x.min(), _x.max(), 100)
        _ax.plot(10 ** _xg, _m * _xg + _b, color=_color, linewidth=2, linestyle=_ls)

    _fit_line(_log_d, _y, _all_trend, "-")
    _fit_line(_log_d[~_is_excl.to_numpy()], _y[~_is_excl.to_numpy()], _orange, "--")

    _ax.set_xlabel("Population density (people / km²)")
    _ax.set_ylabel("BJP vote share − state mean (pp)")
    _ax.set_title("BJP vote share deviation from state mean vs population density")
    _ax.grid(True, which="both", alpha=0.25)
    _ax.legend(
        handles=[
            _Line2D([0], [0], marker="o", linestyle="", color=_orange, label="Rest of India"),
            _Line2D([0], [0], marker="o", linestyle="", color=_teal, label=_excl_short),
            _Line2D([0], [0], color=_all_trend, linewidth=2, label="All-India trend"),
            _Line2D([0], [0], color=_orange, linewidth=2, linestyle="--", label=f"Excl. {_excl_short} trend"),
        ],
        loc="best", framealpha=0.9,
    )
    _fig.tight_layout()
    _fig.savefig("urban/bjp_vs_popdensity.png", dpi=200, facecolor=_bg)
    _ax
    return


@app.cell
def _(scatter_df):
    import numpy as _np
    import statsmodels.formula.api as _smf
    from statsmodels.iolib.summary2 import summary_col as _summary_col

    _d = scatter_df.to_pandas()
    _d = _d.dropna(subset=["pop_density", "vote_share_percentage", "state_ut"])
    _d = _d[_d["pop_density"] > 0].copy()
    _d["log_dens"] = _np.log10(_d["pop_density"])

    _m = _smf.ols(
        "vote_share_percentage ~ log_dens + C(state_ut)", data=_d
    ).fit(cov_type="cluster", cov_kwds={"groups": _d["state_ut"]})

    _n_states = _d["state_ut"].nunique()
    _table = _summary_col(
        [_m],
        model_names=["BJP vote share"],
        regressor_order=["log_dens"],
        drop_omitted=True,
        info_dict={
            "N": lambda r: f"{int(r.nobs):,}",
            "States": lambda r: f"{_n_states}",
            "R-squared": lambda r: f"{r.rsquared:.3f}",
            "State FE": lambda r: "Yes",
            "SE": lambda r: "State-clustered",
        },
    )
    _table.title = "BJP vote share vs. population density (AC level, state fixed effects)"

    _latex = _table.as_latex().replace("\\hline", "\\midrule").split("\\end{table}")[0] + "\\end{table}\n"
    with open("urban/bjp_density_results.tex", "w") as _f:
        _f.write(_latex)
    _table
    return


if __name__ == "__main__":
    app.run()
