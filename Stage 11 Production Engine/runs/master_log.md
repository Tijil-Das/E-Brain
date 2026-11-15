### Run `RUN_20261115_002`: Baseline: SGD Online

- **Date/Time:** 2026-11-15 08:31:43
- **Git Commit:** `10c25f34bbfcb3b0c8bc9eca64cbc4599fa589d0`
- **Data Source:** `Text.txt` (85389 chars, SHA-256: `0f12fa1f44473f32`)
- **Model Architecture:** $L=2, H=null, D=32, H_{mlp}=64$, Params: 21508
- **Optimization:** SGD, LR=0.02, Batch=1, Dropout=null
- **Outcome:** Best Epoch **4** | Train Loss: **2.7152** | Best Val Loss (Pos 8..15): **2.7298**

```text
Arthur Pendelton arte tade s hesy of at csend  tegre te iud cherothe asefas bd tiAbe amide terenu iPcanterlune mararhit ted s ten saf, ld ffe s d bd cafe aresa- thaderholt tlalrthe shelrpavthg nd te sthe teserheriinth
```

---

### Run `RUN_20261115_003`: Ablation 1: + Adam Optimizer

- **Date/Time:** 2026-11-15 08:33:14
- **Git Commit:** `10c25f34bbfcb3b0c8bc9eca64cbc4599fa589d0`
- **Data Source:** `Text.txt` (85389 chars, SHA-256: `0f12fa1f44473f32`)
- **Model Architecture:** $L=2, H=null, D=32, H_{mlp}=64$, Params: 21508
- **Optimization:** Adam, LR=0.003, Batch=1, Dropout=null
- **Outcome:** Best Epoch **4** | Train Loss: **2.3782** | Best Val Loss (Pos 8..15): **2.4664**

```text
Arthur Pendelton asdons borendl ethee boteoos carepHe ulo atul derkaise simedur sthe doftor tuohed agu stwhed sa fred sive teacr oleshed blacat dasssthe unthed theal, ssticon soppemp tha tor tcred tarengestustragutbrs
```

---

### Run `RUN_20261115_004`: Ablation 2: + Mini-Batches (B=16)

- **Date/Time:** 2026-11-15 08:34:38
- **Git Commit:** `10c25f34bbfcb3b0c8bc9eca64cbc4599fa589d0`
- **Data Source:** `Text.txt` (85389 chars, SHA-256: `0f12fa1f44473f32`)
- **Model Architecture:** $L=2, H=null, D=32, H_{mlp}=64$, Params: 21508
- **Optimization:** Adam, LR=0.005, Batch=16, Dropout=null
- **Outcome:** Best Epoch **4** | Train Loss: **2.5021** | Best Val Loss (Pos 8..15): **2.5469**

```text
Arthur Pendelton tuwud co tteerad aw ded alheat t thaula as t t f erarilaoft pethel, orit te gomveor ?r tket an netf sethec ras sk sten acoed bicrp theven wots thed heritin ow ton cof w mat d in e thainsowto ig k lada
```

---

### Run `RUN_20261115_005`: Ablation 3: + LR Warmup (1 Epoch)

- **Date/Time:** 2026-11-15 08:36:21
- **Git Commit:** `10c25f34bbfcb3b0c8bc9eca64cbc4599fa589d0`
- **Data Source:** `Text.txt` (85389 chars, SHA-256: `0f12fa1f44473f32`)
- **Model Architecture:** $L=2, H=null, D=32, H_{mlp}=64$, Params: 21508
- **Optimization:** Adam, LR=0.005, Batch=16, Dropout=null
- **Outcome:** Best Epoch **4** | Train Loss: **2.5021** | Best Val Loss (Pos 8..15): **2.5469**

```text
Arthur Pendelton e, sk, an thon bo t thor pine gofor ntk athey s minand tle glalis t. f toud that ile thinid at in g hart, athe s ltathouleer ft da te t on wee fthints sm ban bler y alte me ararvest redy n pe faxb e e
```

---

### Run `RUN_20261115_006`: Ablation 4: + Pre-LayerNorm

- **Date/Time:** 2026-11-15 08:38:04
- **Git Commit:** `10c25f34bbfcb3b0c8bc9eca64cbc4599fa589d0`
- **Data Source:** `Text.txt` (85389 chars, SHA-256: `0f12fa1f44473f32`)
- **Model Architecture:** $L=2, H=null, D=32, H_{mlp}=64$, Params: 21828
- **Optimization:** Adam, LR=0.005, Batch=16, Dropout=null
- **Outcome:** Best Epoch **4** | Train Loss: **2.3640** | Best Val Loss (Pos 8..15): **2.4189**

```text
Arthur Pendelton drad mras the thivy he mot to on ame dar the heing torts atey was at the aseomof enicw ale innceeshat gring the whach flited turt poamtey cuorthe than lor sever ied thon don thelr tharm woo Planthin m
```

---

