from lxml import html

m = html.parse("ceos/map.html")
links = [x.strip() + "\n" for x in m.xpath("//a/@href")]
links = sorted(list(set(links)))
with open("ceos/map_links.txt", "w") as f:
    f.writelines(links)


### extract ceo links from the map displayed at https://www.eci.gov.in
