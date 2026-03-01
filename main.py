"""
main.py
=======
配車最適化 API - FastAPI エントリーポイント + VRP ソルバー + 距離計算

エンドポイント:
  GET  /          ヘルスチェック
  POST /optimize  VRP 最適化実行

Render へのデプロイ:
  1. requirements.txt の依存関係をインストール
  2. 起動コマンド: uvicorn main:app --host 0.0.0.0 --port $PORT
  ※ 環境変数の設定は不要（ジオコーディングは GAS 側で実施）
"""

import math
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# 距離・時間マトリックス計算
# ─────────────────────────────────────────────

def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """2点間の球面距離（km）をハベサイン公式で計算する。"""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlng / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _build_time_matrix(
    locations: list[dict], speed_kmh: float, detour_factor: float
) -> list[list[int]]:
    """ロケーション間の移動時間マトリックス（分・整数）を返す。"""
    n = len(locations)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            km = _haversine_km(
                locations[i]["lat"], locations[i]["lng"],
                locations[j]["lat"], locations[j]["lng"],
            ) * detour_factor
            matrix[i][j] = max(1, int(km / speed_kmh * 60))
    return matrix


def _build_distance_matrix(
    locations: list[dict], detour_factor: float
) -> list[list[float]]:
    """ロケーション間の走行距離マトリックス（km・float）を返す。"""
    n = len(locations)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = _haversine_km(
                    locations[i]["lat"], locations[i]["lng"],
                    locations[j]["lat"], locations[j]["lng"],
                ) * detour_factor
    return matrix


# ─────────────────────────────────────────────
# VRP ソルバー
# ─────────────────────────────────────────────

DEPOT_START_MIN = 8 * 60   # 08:00
DEPOT_END_MIN   = 20 * 60  # 20:00
MAX_WORK_MINS   = DEPOT_END_MIN - DEPOT_START_MIN  # 720分
SOLVE_TIME_SEC  = 30       # OR-Tools 最大求解時間（秒）※件数が多い場合は増やしてよい


def _parse_time(t_str: str) -> int:
    """'HH:MM' → 分（整数）"""
    parts = t_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def _mins_to_str(m: int) -> str:
    """分（整数）→ 'HH:MM'"""
    return f"{m // 60:02d}:{m % 60:02d}"


def _optimize(
    depots: list[dict],
    vehicles: list[dict],
    deliveries: list[dict],
    settings: dict,
) -> tuple[list[dict], float] | None:
    """
    VRP を解いてルートと総走行距離を返す。解なしの場合は None を返す。

    制約: 時間窓・積載量・車両進入制限（小型限定）・作業時間・マルチデポ
    """
    detour_factor  = settings.get("detour_factor", 1.3)
    speed_kmh      = settings.get("speed_kmh", 40.0)
    num_vehicles   = len(vehicles)
    num_deliveries = len(deliveries)

    # ノードリスト: [デポ...] + [顧客...]
    depot_map: dict[int, dict] = {d["id"]: d for d in depots}
    used_depot_ids = sorted({v.get("depot_id", 0) for v in vehicles})

    node_locations: list[dict] = []
    depot_node_idx: dict[int, int] = {}

    for dep_id in used_depot_ids:
        if dep_id in depot_map:
            depot_node_idx[dep_id] = len(node_locations)
            node_locations.append(depot_map[dep_id])
        else:
            depot_node_idx[dep_id] = 0

    customer_node_start = len(node_locations)
    for d in deliveries:
        node_locations.append({"lat": d["lat"], "lng": d["lng"]})

    n_nodes = len(node_locations)
    time_matrix = _build_time_matrix(node_locations, speed_kmh, detour_factor)
    dist_matrix = _build_distance_matrix(node_locations, detour_factor)

    vehicle_depot_nodes = [
        depot_node_idx.get(v.get("depot_id", 0), 0)
        for v in vehicles
    ]

    # OR-Tools モデル構築
    manager = pywrapcp.RoutingIndexManager(
        n_nodes, num_vehicles,
        vehicle_depot_nodes, vehicle_depot_nodes,
    )
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_idx: int, to_idx: int) -> int:
        from_node = manager.IndexToNode(from_idx)
        to_node   = manager.IndexToNode(to_idx)
        svc = 0
        if from_node >= customer_node_start:
            svc = int(deliveries[from_node - customer_node_start].get("service_time_min", 15))
        return time_matrix[from_node][to_node] + svc

    transit_cb = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    routing.AddDimension(transit_cb, 60, MAX_WORK_MINS, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    for i, d in enumerate(deliveries):
        r_idx = manager.NodeToIndex(customer_node_start + i)
        tw_s  = max(0, _parse_time(d.get("time_window_start", "08:00")) - DEPOT_START_MIN)
        tw_e  = max(tw_s + 1, min(
            _parse_time(d.get("time_window_end", "20:00")) - DEPOT_START_MIN,
            MAX_WORK_MINS,
        ))
        time_dim.CumulVar(r_idx).SetRange(tw_s, tw_e)

    for v_idx in range(num_vehicles):
        time_dim.CumulVar(routing.Start(v_idx)).SetRange(0, 0)
        time_dim.CumulVar(routing.End(v_idx)).SetRange(0, MAX_WORK_MINS)

    def demand_callback(from_idx: int) -> int:
        node = manager.IndexToNode(from_idx)
        if node < customer_node_start:
            return 0
        return int(deliveries[node - customer_node_start].get("demand_kg", 0))

    demand_cb = routing.RegisterUnaryTransitCallback(demand_callback)
    capacities = [int(v.get("capacity_kg", 4000)) for v in vehicles]
    routing.AddDimensionWithVehicleCapacity(demand_cb, 0, capacities, True, "Capacity")

    for i, d in enumerate(deliveries):
        if d.get("require_small_truck"):
            r_idx = manager.NodeToIndex(customer_node_start + i)
            for v_idx, v in enumerate(vehicles):
                if v.get("is_8t"):
                    routing.VehicleVar(r_idx).RemoveValue(v_idx)

    penalty = 10_000_000
    for i in range(num_deliveries):
        routing.AddDisjunction([manager.NodeToIndex(customer_node_start + i)], penalty)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = SOLVE_TIME_SEC

    solution = routing.SolveWithParameters(search_params)
    if not solution:
        return None

    routes: list[dict] = []
    total_distance_km = 0.0

    for v_idx, v in enumerate(vehicles):
        index = routing.Start(v_idx)
        stops = []
        order = 1

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node >= customer_node_start:
                d = deliveries[node - customer_node_start]
                arrival_min = DEPOT_START_MIN + solution.Value(time_dim.CumulVar(index))
                stops.append({
                    "delivery_idx": d["idx"],
                    "route_order":  order,
                    "eta":          _mins_to_str(arrival_min),
                    "demand_kg":    d.get("demand_kg", 0),
                })
                order += 1
            next_index = solution.Value(routing.NextVar(index))
            total_distance_km += dist_matrix[node][manager.IndexToNode(next_index)]
            index = next_index

        if stops:
            routes.append({"vehicle_id": v["vehicle_id"], "stops": stops})

    return routes, total_distance_km


# ─────────────────────────────────────────────
# FastAPI アプリ
# ─────────────────────────────────────────────

app = FastAPI(
    title="配車最適化 API",
    description="OR-Tools VRP ソルバー（時間窓・積載量・進入制限対応）",
    version="1.0.0",
)


class DepotModel(BaseModel):
    id: int
    lat: float
    lng: float


class VehicleModel(BaseModel):
    vehicle_id:   int
    vehicle_name: Optional[str] = ""
    capacity_kg:  int
    is_8t:        int = 0
    depot_id:     int = 0


class DeliveryModel(BaseModel):
    idx:                 int
    customer_name:       Optional[str] = ""
    address:             Optional[str] = ""
    lat:                 float
    lng:                 float
    require_small_truck: int   = 0
    service_time_min:    int   = 15
    demand_kg:           int   = 0
    time_window_start:   str   = "08:00"
    time_window_end:     str   = "20:00"


class SettingsModel(BaseModel):
    detour_factor: float = Field(default=1.3, ge=1.0, le=3.0)
    speed_kmh:     float = Field(default=40.0, gt=0)


class OptimizeRequest(BaseModel):
    depots:     list[DepotModel]
    vehicles:   list[VehicleModel]
    deliveries: list[DeliveryModel]
    settings:   SettingsModel = SettingsModel()


@app.get("/", summary="ヘルスチェック")
def root():
    return {"status": "ok", "service": "配車最適化 API"}


@app.post("/optimize", summary="VRP 最適化実行")
def optimize_route(req: OptimizeRequest):
    """
    GAS から受け取ったデポ・車両・配送データを OR-Tools で最適化し、
    各配送先の割当車両・配送順序・到着予想時刻（ETA）を返す。
    ジオコーディングは GAS 側で実施済みの前提（lat/lng が設定されていること）。
    """
    if not req.deliveries:
        raise HTTPException(status_code=400, detail="deliveries が空です")
    if not req.vehicles:
        raise HTTPException(status_code=400, detail="vehicles が空です")
    if not req.depots:
        raise HTTPException(status_code=400, detail="depots が空です")

    # lat/lng が未設定の配送先を検出して警告
    missing = [d.idx for d in req.deliveries if not d.lat and not d.lng]
    if missing:
        return {
            "status":  "error",
            "message": f"緯度経度が未設定の配送先があります（idx: {missing}）。住所を入力してジオコーディングを完了させてください。",
        }

    start_ms = time.time()

    result = _optimize(
        depots     = [d.model_dump() for d in req.depots],
        vehicles   = [v.model_dump() for v in req.vehicles],
        deliveries = [d.model_dump() for d in req.deliveries],
        settings   = req.settings.model_dump(),
    )

    elapsed_ms = int((time.time() - start_ms) * 1000)

    if result is None:
        return {
            "status":  "error",
            "message": "最適解が見つかりませんでした。データを確認してください。",
        }

    routes, total_distance_km = result
    return {
        "status":            "success",
        "routes":            routes,
        "total_distance_km": round(total_distance_km, 2),
        "solve_time_ms":     elapsed_ms,
    }
