import pickle, numpy as np
from pathlib import Path

res = pickle.load(open("inference_result_np_large_arbon_events_evening_copy_softplus.pickle","rb"))
dp = Path(res["data_pickle"])
if not dp.exists(): dp = Path(".")/dp.name
data = pickle.load(open(dp,"rb"))

node = np.asarray(data.get("node_locations", []))
print("node_locations shape:", node.shape)
if node.size: print("node first5:\n", node[:5], "\nmin/max:", node.min(0), node.max(0))

ev = data["events"]
if isinstance(ev, dict):
    for k in ("x","y"):
        if k in ev:
            a = np.asarray(ev[k]); print(f"{k} shape {a.shape} min {a.min()} max {a.max()}")