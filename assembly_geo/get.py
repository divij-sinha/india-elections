import urllib
import urllib.parse

import httpx

old_base = "https://livingatlas.esri.in/server/rest/services/Election_Legislative_Assembly_2022/MapServer/0"

base_url = """https://livingatlas.esri.in/server/rest/services/IAB2024/IN_Legislative_Assembly_2024/MapServer/0/query?f=json&geometry=%7B%22xmin%22%3A7514065.628541023%2C%22ymin%22%3A0.000002983957529067993%2C%22xmax%22%3A10800000.000%2C%22ymax%22%3A4552779.631730943%7D&orderByFields=objectid&outFields={out_fields}&outSR=102100&resultType=tile&returnCentroid=true&returnExceededLimitFeatures=true&spatialRel=esriSpatialRelIntersects&where={where_clause}&geometryType=esriGeometryEnvelope&inSR=102100"""
session = httpx.Client(http2=True)

for i in range(0, 5000, 1000):
    out_fields = urllib.parse.quote(
        (
            "objectid,win_prty,cons_name,elec_year,elec_date,no_seat,elec_state,ball_state,votes_state,no_cndt,parties_st"
            ",ind_cndt,cons_code,seat_cons,sc_caste,st_tribe,cndt_cons,cndt_cons_f,elec_cons,ball_cons,votes_cons,win_votes"
            ",sec_votes,trd_votes,frt_votes,fif_votes,six_votes,other_cndt,other_vote,st_area(shape),st_length(shape)"
        )
    )
    where_clause = urllib.parse.quote(f"objectid>{i}AND objectid<={i+1000}")
    cur_url = base_url.format(out_fields=out_fields, where_clause=where_clause)
    resp = session.get(cur_url)
    if resp.status_code == 200 and (
        ("application/json" in resp.headers.get("content-type"))
        or ("application/json" in resp.headers.get("Content-Type"))
    ):
        with open("data/assembly_geo_{}.json".format(i), "w") as f:
            f.write(resp.text)

LINK_1000 = """https://livingatlas.esri.in/server/rest/services/IAB2024/IN_Legislative_Assembly_2024/MapServer/0/query?f=json&geometry=%7B%22xmin%22%3A7514065.628541023%2C%22ymin%22%3A0.000002983957529067993%2C%22xmax%22%3A10800000.000%2C%22ymax%22%3A4552779.631730943%7D&orderByFields=objectid&outFields=objectid%2Cwin_prty&outSR=102100&resultType=tile&returnCentroid=true&returnExceededLimitFeatures=true&spatialRel=esriSpatialRelIntersects&where="objectid%3C1000"&geometryType=esriGeometryEnvelope&inSR=102100"""
LINK_2000 = """https://livingatlas.esri.in/server/rest/services/IAB2024/IN_Legislative_Assembly_2024/MapServer/0/query?f=json&geometry=%7B%22xmin%22%3A7514065.628541023%2C%22ymin%22%3A0.000002983957529067993%2C%22xmax%22%3A10800000.000%2C%22ymax%22%3A4552779.631730943%7D&orderByFields=objectid&outFields=objectid%2Cwin_prty&outSR=102100&resultType=tile&returnCentroid=true&returnExceededLimitFeatures=true&spatialRel=esriSpatialRelIntersects&where="objectid%3E999%20AND%20objectid%3C2000"&geometryType=esriGeometryEnvelope&inSR=102100"""
LINK_3000 = """https://livingatlas.esri.in/server/rest/services/IAB2024/IN_Legislative_Assembly_2024/MapServer/0/query?f=json&geometry=%7B%22xmin%22%3A7514065.628541023%2C%22ymin%22%3A0.000002983957529067993%2C%22xmax%22%3A10800000.000%2C%22ymax%22%3A4552779.631730943%7D&orderByFields=objectid&outFields=objectid%2Cwin_prty&outSR=102100&resultType=tile&returnCentroid=true&returnExceededLimitFeatures=true&spatialRel=esriSpatialRelIntersects&where="objectid%3E1999%20AND%20objectid%3C3000"&geometryType=esriGeometryEnvelope&inSR=102100"""
LINK_4000 = """https://livingatlas.esri.in/server/rest/services/IAB2024/IN_Legislative_Assembly_2024/MapServer/0/query?f=json&geometry=%7B%22xmin%22%3A7514065.628541023%2C%22ymin%22%3A0.000002983957529067993%2C%22xmax%22%3A10800000.000%2C%22ymax%22%3A4552779.631730943%7D&orderByFields=objectid&outFields=objectid%2Cwin_prty&outSR=102100&resultType=tile&returnCentroid=true&returnExceededLimitFeatures=true&spatialRel=esriSpatialRelIntersects&where="objectid%3E2999%20AND%20objectid%3C4000"&geometryType=esriGeometryEnvelope&inSR=102100"""
LINK_5000 = """https://livingatlas.esri.in/server/rest/services/IAB2024/IN_Legislative_Assembly_2024/MapServer/0/query?f=json&geometry=%7B%22xmin%22%3A7514065.628541023%2C%22ymin%22%3A0.000002983957529067993%2C%22xmax%22%3A10800000.000%2C%22ymax%22%3A4552779.631730943%7D&orderByFields=objectid&outFields=objectid%2Cwin_prty&outSR=102100&resultType=tile&returnCentroid=true&returnExceededLimitFeatures=true&spatialRel=esriSpatialRelIntersects&where="objectid%3E3999"&geometryType=esriGeometryEnvelope&inSR=102100"""
# LINK_1000 = """https://livingatlas.esri.in/server/rest/services/IAB2024/IN_Legislative_Assembly_2024/MapServer/0/query?f=json&geometry=%7B%22xmin%22%3A7514065.628541023%2C%22ymin%22%3A0.000002983957529067993%2C%22xmax%22%3A10800000.000%2C%22ymax%22%3A4552779.631730943%7D&orderByFields=objectid&outFields=objectid%2Cwin_prty&outSR=102100&resultType=tile&returnCentroid=true&returnExceededLimitFeatures=true&spatialRel=esriSpatialRelIntersects&where="objectid%3C1000"&geometryType=esriGeometryEnvelope&inSR=102100"""
# LINK_2000 = """https://livingatlas.esri.in/server/rest/services/IAB2024/IN_Legislative_Assembly_2024/MapServer/0/query?f=json&geometry=%7B%22xmin%22%3A7514065.628541023%2C%22ymin%22%3A0.000002983957529067993%2C%22xmax%22%3A10800000.000%2C%22ymax%22%3A4552779.631730943%7D&orderByFields=objectid&outFields=objectid%2Cwin_prty&outSR=102100&resultType=tile&returnCentroid=true&returnExceededLimitFeatures=true&spatialRel=esriSpatialRelIntersects&where="objectid%3E999%20AND%20objectid%3C2000"&geometryType=esriGeometryEnvelope&inSR=102100"""
# LINK_3000 = """https://livingatlas.esri.in/server/rest/services/IAB2024/IN_Legislative_Assembly_2024/MapServer/0/query?f=json&geometry=%7B%22xmin%22%3A7514065.628541023%2C%22ymin%22%3A0.000002983957529067993%2C%22xmax%22%3A10800000.000%2C%22ymax%22%3A4552779.631730943%7D&orderByFields=objectid&outFields=objectid%2Cwin_prty&outSR=102100&resultType=tile&returnCentroid=true&returnExceededLimitFeatures=true&spatialRel=esriSpatialRelIntersects&where="objectid%3E1999%20AND%20objectid%3C3000"&geometryType=esriGeometryEnvelope&inSR=102100"""
# LINK_4000 = """https://livingatlas.esri.in/server/rest/services/IAB2024/IN_Legislative_Assembly_2024/MapServer/0/query?f=json&geometry=%7B%22xmin%22%3A7514065.628541023%2C%22ymin%22%3A0.000002983957529067993%2C%22xmax%22%3A10800000.000%2C%22ymax%22%3A4552779.631730943%7D&orderByFields=objectid&outFields=objectid%2Cwin_prty&outSR=102100&resultType=tile&returnCentroid=true&returnExceededLimitFeatures=true&spatialRel=esriSpatialRelIntersects&where="objectid%3E2999%20AND%20objectid%3C4000"&geometryType=esriGeometryEnvelope&inSR=102100"""
# LINK_5000 = """https://livingatlas.esri.in/server/rest/services/IAB2024/IN_Legislative_Assembly_2024/MapServer/0/query?f=json&geometry=%7B%22xmin%22%3A7514065.628541023%2C%22ymin%22%3A0.000002983957529067993%2C%22xmax%22%3A10800000.000%2C%22ymax%22%3A4552779.631730943%7D&orderByFields=objectid&outFields=objectid%2Cwin_prty&outSR=102100&resultType=tile&returnCentroid=true&returnExceededLimitFeatures=true&spatialRel=esriSpatialRelIntersects&where="objectid%3E3999"&geometryType=esriGeometryEnvelope&inSR=102100"""

# "https://livingatlas.esri.in/server/rest/services/Election_Legislative_Assembly_2022/MapServer/0"
