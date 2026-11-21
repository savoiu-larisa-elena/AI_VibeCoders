"""Trip layer: where is the car, what is ahead, where should a tired driver stop.

Pure Python, no extra dependencies. REAL data, works offline:
  * data/route_cache.json - real road geometry Cluj-Napoca -> Turda -> Aiud -> Alba Iulia ->
    Sebes -> Sibiu (OSRM, ~180 km, fetched once and saved)
  * data/poi_cache.json   - real places along that road from OpenStreetMap (rest areas,
    motorway services, fuel stations, malls, hotels/guesthouses), fetched once and saved
If those files are missing, a straight-line demo route with made-up demo places is used.

    trip = Trip()                       # real route if data/ exists, else the demo route
    trip.advance(dt_seconds)            # simulated GPS (call every frame)
    rec = Advisor(trip).recommend(level)   # level: 0 ok, 1 tired, 2 very tired
    # rec -> {"kind": "rest_stop", "name": ..., "distance_km": ..., "eta_min": ..., ...}

Key idea: POIs are judged by distance AHEAD ALONG THE ROUTE, not by straight-line radius.
A hotel 3 km away on another road (or behind you) is useless; one 9 km ahead is perfect.
"""
from __future__ import annotations

import json
import math
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path

R_EARTH_KM = 6371.0
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ROUTE_FILE = DATA_DIR / "route_cache.json"
POI_FILE = DATA_DIR / "poi_cache.json"
MIN_AHEAD_KM = 0.3          # closer than this: too late to brake and turn in


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_EARTH_KM * math.asin(math.sqrt(h))


@dataclass
class Poi:
    name: str
    kind: str            # "rest_stop" | "fuel" | "mall" | "hotel"
    lat: float
    lon: float
    note: str = ""


class Route:
    """A polyline with cumulative distances, so we can ask 'where am I after X km'
    and 'how far along the route is this point'."""

    def __init__(self, points: list[tuple[float, float]]):
        self.points = points
        self.cum = [0.0]
        for p, q in zip(points, points[1:]):
            self.cum.append(self.cum[-1] + haversine_km(p, q))
        self.length_km = self.cum[-1]

    def position_at(self, km: float) -> tuple[float, float]:
        km = max(0.0, min(km, self.length_km))
        i = min(max(bisect_right(self.cum, km), 1), len(self.points) - 1)   # fast on long routes
        seg = self.cum[i] - self.cum[i - 1]
        t = (km - self.cum[i - 1]) / seg if seg else 0.0
        a, b = self.points[i - 1], self.points[i]
        return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t

    def project(self, lat: float, lon: float) -> tuple[float, float]:
        """(km along the route, km off the route) of the closest route vertex/segment sample.
        Samples every ~1 km: plenty for 'is it near the road and how far ahead'."""
        if not hasattr(self, "_samples"):                 # sample the route once (every 0.5 km)
            n = int(self.length_km / 0.5) + 2
            self._samples = [(k * 0.5, self.position_at(k * 0.5)) for k in range(n)]
        best_along, best_off = 0.0, 1e9
        for km, pos in self._samples:
            d = haversine_km(pos, (lat, lon))
            if d < best_off:
                best_along, best_off = km, d
        return best_along, best_off


# Demo route through real towns (approximate town-centre coordinates). Later: replace with a
# real road polyline (OSRM / GPX export) - the rest of the code does not change.
DEMO_ROUTE = [
    (46.770, 23.590),  # Cluj-Napoca
    (46.570, 23.780),  # Turda
    (46.540, 24.560),  # Targu Mures
    (46.220, 24.790),  # Sighisoara
    (45.650, 25.610),  # Brasov
]

# OFFLINE DEMO POIs, placed exactly ON the demo route at fixed km (so the demo always has an
# answer: a pit stop every ~25 km, lodging at least every ~60 km). Names are generic on purpose:
# these are NOT real businesses and the positions are approximate. Step 5 replaces/extends them
# with real OpenStreetMap places.
_DEMO_KM = [  # (km along route, kind, name, note)
    (2, "hotel", "Lodging (demo) Cluj-Napoca", "hotel in town"),
    (14, "rest_stop", "Rest area (demo) before Turda", "parking, toilets"),
    (28, "fuel", "Fuel + cafe (demo) Turda", "coffee, open 24/7"),
    (28, "hotel", "Lodging (demo) Turda", "guesthouse"),
    (50, "rest_stop", "Rest area (demo) Campia Turzii", "parking, shade"),
    (70, "fuel", "Fuel + cafe (demo) Sarateni", "coffee, open 24/7"),
    (86, "rest_stop", "Rest area (demo) Targu Mures", "parking, toilets"),
    (86, "hotel", "Lodging (demo) Targu Mures", "hotel in town"),
    (108, "fuel", "Fuel + cafe (demo) Targu Mures - Sighisoara", "coffee, open 24/7"),
    (126, "rest_stop", "Rest area (demo) Sighisoara", "parking"),
    (126, "hotel", "Lodging (demo) Sighisoara", "guesthouse, reception"),
    (150, "fuel", "Fuel + cafe (demo) Sighisoara - Rupea", "coffee, open 24/7"),
    (172, "rest_stop", "Rest area (demo) Rupea", "parking"),
    (172, "hotel", "Guesthouse (demo) Rupea", "guesthouse"),
    (195, "fuel", "Fuel + cafe (demo) near Brasov", "coffee, open 24/7"),
    (214, "hotel", "Lodging (demo) Brasov", "hotel in town"),
]


def _build_demo_pois() -> list[Poi]:
    route = Route(DEMO_ROUTE)
    out = []
    for km, kind, name, note in _DEMO_KM:
        lat, lon = route.position_at(km)
        out.append(Poi(name, kind, round(lat, 5), round(lon, 5), note))
    return out


DEMO_POIS = _build_demo_pois()


def distance_text(km: float) -> str:
    """Spoken-friendly distance: '300 meters', '4.5 kilometers', '12 kilometers'."""
    m = km * 1000
    if m < 1000:
        m = max(50, int(round(m / 50.0)) * 50)
        return f"{m} meters" if m < 1000 else "1 kilometer"
    return f"{km:.1f} kilometers" if km < 10 else f"{round(km)} kilometers"


def load_real_route() -> "Route | None":
    try:
        pts = json.loads(ROUTE_FILE.read_text(encoding="utf-8"))["points"]
        return Route([(float(a), float(b)) for a, b in pts])
    except (OSError, ValueError, KeyError):
        return None


def load_real_pois() -> "list[Poi] | None":
    try:
        places = json.loads(POI_FILE.read_text(encoding="utf-8"))["places"]
    except (OSError, ValueError, KeyError):
        return None
    defaults = {"fuel": "Fuel station", "hotel": "Hotel", "mall": "Shopping mall",
                "rest_stop": "Rest area"}
    out = []
    for p in places:
        name = str(p.get("name") or "").strip()
        if len(name) < 3 or name.isdigit():           # e.g. a fuel station named just '98'
            name = defaults.get(p["kind"], "Stop")
        out.append(Poi(name, p["kind"], float(p["lat"]), float(p["lon"]), p.get("note", "")))
    return out or None


def _demo_pois_for(route: "Route") -> "list[Poi]":
    """Offline fallback for a real route without the places file: generic demo places
    every ~20 km (pit stops) plus some lodging. Made up - the names say '(demo)'."""
    out, km = [], 8.0
    while km < route.length_km - 2:
        lat, lon = route.position_at(km)
        out.append(Poi(f"Rest area (demo) km {km:.0f}", "rest_stop", round(lat, 5), round(lon, 5), "parking"))
        if int(km) % 45 < 20:
            out.append(Poi(f"Lodging (demo) km {km:.0f}", "hotel", round(lat, 5), round(lon, 5), "hotel"))
        km += 20.0
    return out


class Trip:
    """Simulated vehicle moving along a route. `speed_factor` > 1 speeds the demo up
    (e.g. 20 = a 100 km/h car covers ~0.55 km per real second)."""

    def __init__(self, route: Route | None = None, pois: list[Poi] | None = None,
                 speed_kmh: float = 100.0, speed_factor: float = 1.0):
        real_route = None if route else load_real_route()
        real_pois = load_real_pois() if (pois is None and real_route) else None
        self.route = route or real_route or Route(DEMO_ROUTE)
        if pois is not None:
            self.pois = pois
        elif real_pois:
            self.pois = real_pois
        else:                                    # offline fallback: demo places on this route
            self.pois = list(DEMO_POIS) if not real_route else _demo_pois_for(self.route)
        if real_route and real_pois:
            self.data_source = "real road (OSRM) + OpenStreetMap places"
        elif real_route:
            self.data_source = "real road (OSRM) + demo places"
        else:
            self.data_source = "demo route + demo places"
        self.route_name = "Cluj-Napoca - Sibiu" if real_route else "Cluj-Napoca - Brasov (demo)"
        self.speed_kmh = speed_kmh
        self.speed_factor = speed_factor
        self.km = 0.0
        self._proj: dict[tuple[float, float], tuple[float, float]] = {}   # POI -> (along km, off km)

    def advance(self, dt_seconds: float) -> None:
        self.km = min(self.route.length_km, self.km + self.speed_kmh * self.speed_factor * dt_seconds / 3600)

    @property
    def position(self) -> tuple[float, float]:
        return self.route.position_at(self.km)

    def ahead(self, kinds: tuple[str, ...], max_km: float, max_off_route_km: float = 3.0) -> list[dict]:
        """POIs of the given kinds that are AHEAD on the route within max_km, nearest first."""
        out = []
        for p in self.pois:
            if p.kind not in kinds:
                continue
            key = (p.lat, p.lon)
            if key not in self._proj:               # a POI never moves: project it once
                self._proj[key] = self.route.project(p.lat, p.lon)
            along, off = self._proj[key]
            dist = along - self.km
            # at least MIN_AHEAD_KM away: a place 50 m ahead can't be reached safely any more
            if MIN_AHEAD_KM <= dist <= max_km and off <= max_off_route_km:
                out.append({**asdict(p), "distance_km": round(dist, 1),
                            "distance_text": distance_text(dist),
                            "eta_min": round(dist / self.speed_kmh * 60, 1)})
        return sorted(out, key=lambda r: r["distance_km"])


class Advisor:
    """Maps a fatigue level to a concrete stop.

    level 0 -> nothing
    level 1 (tired)      -> a pit stop (rest area / fuel+coffee) within `pit_stop_km`
    level 2 (very tired) -> a hotel within `hotel_km`; if none is close enough, the nearest
                            safe place to stop NOW (rest area) with urgent wording
    The search radius shrinks as fatigue grows: the sleepier you are, the less distance
    you can be trusted to drive."""

    def __init__(self, trip: Trip, pit_stop_km: float = 40.0, hotel_km: float = 60.0,
                 urgent_stop_km: float = 15.0):
        self.trip = trip
        self.pit_stop_km, self.hotel_km, self.urgent_stop_km = pit_stop_km, hotel_km, urgent_stop_km

    def recommend(self, level: int) -> dict | None:
        if level <= 0:
            return None
        if level == 1:
            found = self.trip.ahead(("rest_stop", "fuel", "mall"), self.pit_stop_km)
            return self._pack("pit_stop", found, urgent=False)
        hotels = self.trip.ahead(("hotel",), self.hotel_km)
        if hotels and hotels[0]["distance_km"] <= self.hotel_km:
            return self._pack("lodging", hotels, urgent=True)
        # no hotel in reach: stop at the nearest safe place now, sleep later
        found = self.trip.ahead(("rest_stop", "fuel", "mall"), self.urgent_stop_km)
        return self._pack("stop_now", found, urgent=True)

    @staticmethod
    def _pack(kind: str, found: list[dict], urgent: bool) -> dict | None:
        if not found:
            return {"kind": kind, "urgent": urgent, "target": None, "alternatives": []}
        # OpenStreetMap sometimes lists one place twice (building + point): one entry per name
        unique, seen = [], set()
        for f in found:
            if f["name"] not in seen:
                seen.add(f["name"])
                unique.append(f)
        return {"kind": kind, "urgent": urgent, "target": unique[0], "alternatives": unique[1:3]}


if __name__ == "__main__":
    trip = Trip()
    adv = Advisor(trip)
    print(f"route length: {trip.route.length_km:.0f} km")
    for km in (5, 60, 110, 180):
        trip.km = km
        for lvl in (1, 2):
            rec = adv.recommend(lvl)
            t = rec["target"] if rec else None
            print(f"km {km:3} level {lvl}: {rec['kind'] if rec else None:9} -> "
                  f"{(t['name'] + ' in ' + str(t['distance_km']) + ' km') if t else 'nothing in reach'}")
