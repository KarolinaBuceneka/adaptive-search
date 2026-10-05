"""
L4_lietotaju_modelis_ADAPTED.py
Lietotāja adaptīvā interfeisa programmatūra.

Sistēma: adaptīva gatavošanas procesu un taimeru sistēma.

Pazīmes:
    x1 - paralēlie taimeri
    x2 - korekciju īpatsvars
    x3 - korekcijas apjoms
    x4 - secību atkārtošana

Palaišana:
    python L4_lietotaju_modelis_ADAPTED.py --generate
"""

import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

SEED = 42
DATA_FILE = Path(__file__).with_name("N2_taimeru_dati.csv")


# ---------------------------------------------------------------------------
# 1. Sintētiskā žurnāla ģenerēšana
# ---------------------------------------------------------------------------

# Katrai grupai: lietotāju skaits un uzvedības parametri.
#   timers             tipiskais vienlaikus aktīvo taimeru skaits
#   p_correction       varbūtība, ka taimeris tiks koriģēts
#   correction_minutes tipiskais korekcijas apjoms minūtēs
#   p_repeat           varbūtība atkārtoti izmantot soļu secību
GENERATE_GROUPS = {
    "majas_pavars": dict(n=10, timers=2, p_correction=0.50, correction_minutes=2, p_repeat=0.20),
    "iesacejs_bez_korekcijam": dict(n=10, timers=2, p_correction=0.10, correction_minutes=1, p_repeat=0.20),
    "profesionalais_pavars": dict(n=10, timers=4, p_correction=0.20, correction_minutes=1, p_repeat=0.70),
}

SESSIONS_PER_USER = 4
PROCESSES_PER_SESSION = 5


def generate_log(path, seed=SEED):
    """Ģenerē sintētisku gatavošanas procesu un taimeru žurnālu."""
    rnd = random.Random(seed)
    rows = []
    uid = 0

    for group, g in GENERATE_GROUPS.items():
        for _ in range(g["n"]):
            uid += 1
            user = f"U{uid:02d}"
            ts = 1_788_000_000 + uid * 100_000

            # Katram lietotājam neliela individuāla novirze no grupas vidējā.
            user_timers = max(1.0, g["timers"] * rnd.uniform(0.75, 1.25))
            user_p_correction = min(1.0, max(0.0, g["p_correction"] * rnd.uniform(0.75, 1.25)))
            user_correction_minutes = max(0.5, g["correction_minutes"] * rnd.uniform(0.75, 1.25))
            user_p_repeat = min(1.0, max(0.0, g["p_repeat"] * rnd.uniform(0.75, 1.25)))

            for s in range(1, SESSIONS_PER_USER + 1):
                session = f"{user}_S{s}"

                for p in range(1, PROCESSES_PER_SESSION + 1):
                    process = f"{session}_P{p}"

                    # Cik taimeri šajā procesā ir aktīvi vienlaikus.
                    active_timers = max(1, min(6, round(rnd.gauss(user_timers, 0.8))))

                    # Vai tiek atkārtoti izmantota iepriekšēja soļu secība.
                    sequence_repeated = 1 if rnd.random() < user_p_repeat else 0
                    sequence_id = "repeated_sequence" if sequence_repeated else f"new_{process}"

                    for timer in range(1, active_timers + 1):
                        timer_id = f"{process}_T{timer}"
                        ts += rnd.randint(20, 90)

                        initial_duration = rnd.randint(5, 30)
                        modified = rnd.random() < user_p_correction
                        correction = 0

                        if modified:
                            correction = max(1, round(rnd.gauss(user_correction_minutes, 0.5)))
                            if rnd.random() < 0.5:
                                correction *= -1

                        final_duration = max(1, initial_duration + correction)

                        rows.append({
                            "user_id": user,
                            "session_id": session,
                            "process_id": process,
                            "timestamp": ts,
                            "timer_id": timer_id,
                            "initial_duration": initial_duration,
                            "final_duration": final_duration,
                            "correction_minutes": correction,
                            "active_timers": active_timers,
                            "sequence_id": sequence_id,
                            "sequence_repeated": sequence_repeated,
                        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "user_id", "session_id", "process_id", "timestamp", "timer_id",
            "initial_duration", "final_duration", "correction_minutes",
            "active_timers", "sequence_id", "sequence_repeated"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def read_log(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# 2. Pazīmju aprēķins
# ---------------------------------------------------------------------------

FEATURES = [
    ("Vidējais paralēlo taimeru skaits", "x1", +1),
    ("Korekciju īpatsvars", "x2", -1),
    ("Vidējais korekcijas apjoms, min", "x3", -1),
    ("Atkārtoto secību īpatsvars", "x4", +1),
]


def compute_features(events):
    """Atgriež x1, x2, x3, x4 vienam lietotājam vai vienai sesijai."""
    if not events:
        return [0.0, 0.0, 0.0, 0.0]

    # Vienam procesam var būt vairākas taimera rindas.
    processes = {}
    for e in events:
        process_id = e["process_id"]
        if process_id not in processes:
            processes[process_id] = {
                "active_timers": int(e["active_timers"]),
                "sequence_repeated": int(e["sequence_repeated"]),
            }

    # x1 - vidējais vienlaikus aktīvo taimeru skaits.
    x1 = sum(p["active_timers"] for p in processes.values()) / len(processes)

    # x2 - koriģēto taimeru īpatsvars.
    corrected = [e for e in events if float(e["correction_minutes"]) != 0]
    x2 = len(corrected) / len(events)

    # x3 - vidējais korekcijas apjoms tikai koriģētajiem taimeriem.
    x3 = (
        sum(abs(float(e["correction_minutes"])) for e in corrected) / len(corrected)
        if corrected else 0.0
    )

    # x4 - procesu īpatsvars, kuros atkārtoti izmantota soļu secība.
    x4 = sum(p["sequence_repeated"] for p in processes.values()) / len(processes)

    return [x1, x2, x3, x4]


def features_by_user(log):
    per_user = defaultdict(list)
    for e in log:
        per_user[e["user_id"]].append(e)
    return {u: compute_features(ev) for u, ev in sorted(per_user.items())}


def features_by_session(log, user):
    per_session = defaultdict(list)
    for e in log:
        if e["user_id"] == user:
            per_session[e["session_id"]].append(e)
    return [compute_features(ev) for _, ev in sorted(per_session.items())]


# ---------------------------------------------------------------------------
# 3. Normalizācija
# ---------------------------------------------------------------------------

def minmax_params(vectors):
    cols = list(zip(*vectors))
    return [min(c) for c in cols], [max(c) for c in cols]


def normalize(x, mins, maxs):
    """Min-max normalizācija; vērtības ārpus diapazona ierobežo līdz [0, 1]."""
    out = []
    for v, lo, hi in zip(x, mins, maxs):
        z = (v - lo) / (hi - lo) if hi > lo else 0.0
        out.append(min(1.0, max(0.0, z)))
    return out


# ---------------------------------------------------------------------------
# 4. Stereotipi un k-vidējo
# ---------------------------------------------------------------------------

def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# Prototipi normalizētajā telpā, pazīmju secība x1, x2, x3, x4.
STEREOTYPES = {
    "Mājas pavārs": [0.0, 1.0, 1.0, 0.0],
    "Iesācējs bez korekcijām": [0.0, 0.0, 0.0, 0.0],
    "Profesionāls pavārs": [1.0, 0.0, 0.0, 1.0],
}


def nearest(x, centers):
    """Atgriež tuvākā centra nosaukumu (vai indeksu) un attālumu."""
    items = centers.items() if isinstance(centers, dict) else enumerate(centers)
    return min(((k, dist(x, c)) for k, c in items), key=lambda t: t[1])


def kmeans(points, k, rnd, max_iter=100):
    centers = [list(p) for p in rnd.sample(points, k)]
    labels = [0] * len(points)

    for iteration in range(max_iter):
        new = [nearest(p, centers)[0] for p in points]
        if new == labels and iteration > 0:
            break
        labels = new

        for j in range(k):
            members = [p for p, l in zip(points, labels) if l == j]
            if members:
                centers[j] = [sum(c) / len(members) for c in zip(*members)]

    inertia = sum(dist(p, centers[l]) ** 2 for p, l in zip(points, labels))
    return labels, centers, inertia


def best_kmeans(points, k, runs=10, seed=SEED):
    """k-vidējo ar `runs` dažādiem sākuma centriem."""
    rnd = random.Random(seed + k)
    return min((kmeans(points, k, rnd) for _ in range(runs)), key=lambda r: r[2])


def silhouette(points, labels):
    k = max(labels) + 1
    scores = []

    for i, p in enumerate(points):
        own = [dist(p, q) for j, q in enumerate(points)
               if labels[j] == labels[i] and j != i]

        if not own:
            scores.append(0.0)
            continue

        a = sum(own) / len(own)
        b = min(
            sum(dist(p, q) for j, q in enumerate(points) if labels[j] == c)
            / max(1, sum(1 for l in labels if l == c))
            for c in range(k) if c != labels[i]
        )
        scores.append((b - a) / max(a, b))

    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# 5. Modelis laikā: EMA un histerēze
# ---------------------------------------------------------------------------

def ema(values, alpha):
    m = values[0]
    out = [m]

    for x in values[1:]:
        m = alpha * x + (1 - alpha) * m
        out.append(m)

    return out


def hysteresis(session_vectors, centers, start, delta=0.1, n_required=2):
    """Klase mainās tikai tad, ja jaunā klase ir tuvāka vismaz par delta N sesijas pēc kārtas."""
    current, candidate, streak, history = start, None, 0, []

    for x in session_vectors:
        best, d_best = nearest(x, centers)
        d_cur = dist(x, centers[current])

        if best != current and d_best < d_cur - delta:
            streak = streak + 1 if best == candidate else 1
            candidate = best

            if streak >= n_required:
                current, candidate, streak = best, None, 0
        else:
            candidate, streak = None, 0

        history.append((best, current))

    return history


# ---------------------------------------------------------------------------
# Izvade
# ---------------------------------------------------------------------------

def fmt(v, d=2):
    return f"{v:.{d}f}".replace(".", ",")


def table(header, rows):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths))
    print(line(header))

    for r in rows:
        print(line(r))

    print()


def main():
    if "--generate" in sys.argv or not DATA_FILE.exists():
        n = generate_log(DATA_FILE)
        print(f"Ģenerēts žurnāls {DATA_FILE.name}: {n} ieraksti, seed = {SEED}\n")

    log = read_log(DATA_FILE)
    feats = features_by_user(log)
    users = list(feats)
    sessions = {e["session_id"] for e in log}

    print(f"Žurnāls: {len(log)} ieraksti, {len(users)} lietotāji, {len(sessions)} sesijas\n")

    short = [f[1] for f in FEATURES]

    print("== Pazīmes (pirmie 5 lietotāji) ==")
    table(["U"] + short,
          [[u] + [fmt(v, 3) for v in feats[u]] for u in users[:5]])

    mins, maxs = minmax_params(list(feats.values()))

    print("== Normalizācijas parametri (min-max) ==")
    table(["pazīme", "min", "max"],
          [[s, fmt(a, 3), fmt(b, 3)] for s, a, b in zip(short, mins, maxs)])

    X = {u: normalize(v, mins, maxs) for u, v in feats.items()}
    points = [X[u] for u in users]

    print("== Stereotipi ==")
    stereo = {u: nearest(X[u], STEREOTYPES)[0] for u in users}

    for name in STEREOTYPES:
        print(f"{name}: {sum(1 for c in stereo.values() if c == name)} lietotāji")

    print()

    print("== k-vidējo (10 palaišanas katram k) ==")
    results = {}
    rows = []

    for k in (2, 3, 4):
        labels, centers, inertia = best_kmeans(points, k)
        s = silhouette(points, labels)
        results[k] = (labels, centers, s, inertia)
        rows.append([k, fmt(s, 3), fmt(inertia, 2)])

    table(["k", "silueta koef.", "iekšējā novirze"], rows)

    # Izvēlamies k ar lielāko silueta koeficientu.
    k = max(results, key=lambda value: results[value][2])
    labels, centers, selected_silhouette, _ = results[k]

    print(f"== Izvēlētās klases, k = {k} ==")

    cluster_names = {j: f"Klase {j + 1}" for j in range(k)}
    named_centers = {cluster_names[j]: c for j, c in enumerate(centers)}

    table(["klase", "n"] + short,
          [[cluster_names[j], labels.count(j)] + [fmt(v) for v in c]
           for j, c in enumerate(centers)])

    print("== Stereotipi pret k-vidējo ==")
    table(["U", "stereotips", "k-vidējo"],
          [[u, stereo[u], cluster_names[l]] for u, l in zip(users, labels)])

    # -----------------------------------------------------------------------
    # 5 jaunu lietotāju klasificēšana
    # -----------------------------------------------------------------------

    print("== Jaunu lietotāju klasificēšana ==")

    rnd = random.Random(SEED + 100)
    test_groups = [
        "iesacejs_bez_korekcijam",
        "profesionalais_pavars",
        "majas_pavars",
        "profesionalais_pavars",
        "majas_pavars",
    ]

    test_rows = []

    for i, group in enumerate(test_groups, 1):
        g = GENERATE_GROUPS[group]

        x = [
            max(1.0, rnd.gauss(g["timers"], 0.8)),
            min(1.0, max(0.0, rnd.gauss(g["p_correction"], 0.08))),
            max(0.0, rnd.gauss(g["correction_minutes"], 0.5)),
            min(1.0, max(0.0, rnd.gauss(g["p_repeat"], 0.08))),
        ]

        cls, d = nearest(normalize(x, mins, maxs), named_centers)
        test_rows.append([f"T{i:02d}"] + [fmt(v, 3) for v in x] + [cls, fmt(d, 3)])

    table(["T"] + short + ["klase", "attālums"], test_rows)

    # -----------------------------------------------------------------------
    # EMA
    # -----------------------------------------------------------------------

    user = users[0]
    per_session = features_by_session(log, user)
    correction_rates = [s[1] for s in per_session]

    print(f"== EMA pazīmei x2, lietotājs {user} ==")

    e3 = ema(correction_rates, 0.3)
    e6 = ema(correction_rates, 0.6)

    table(["sesija", "novērojums", "α = 0,3", "α = 0,6"],
          [[i + 1, fmt(o, 3), fmt(a, 3), fmt(b, 3)]
           for i, (o, a, b) in enumerate(zip(correction_rates, e3, e6))])

    # -----------------------------------------------------------------------
    # Histerēze
    # -----------------------------------------------------------------------

    print("== Histerēze δ = 0,1, N = 2 ==")

    def changes(u):
        near = [
            nearest(normalize(v, mins, maxs), named_centers)[0]
            for v in features_by_session(log, u)
        ]
        return sum(a != b for a, b in zip(near, near[1:]))

    border = max(users, key=changes)
    sess = [normalize(v, mins, maxs) for v in features_by_session(log, border)]
    start = nearest(sess[0], named_centers)[0]
    hist = hysteresis(sess, named_centers, start)

    table(["sesija", "tuvākā klase", "piešķirtā klase"],
          [[i + 1, b, c] for i, (b, c) in enumerate(hist)])

    print(f"Lietotājs {border}, sākuma klase {start}.")


if __name__ == "__main__":
    main()