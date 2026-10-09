# The Master Cheat Sheet — Full AI/ML/GenAI Series
### Math that matters. Production code logic. Edge cases. How to make it better. Nothing else.

---
---

# PART 1 — CLASSICAL ML

---

## 1. Linear & Logistic Regression

**Math that matters:**
```
Linear:    ŷ = w·x + b
Logistic:  ŷ = sigmoid(w·x + b) = 1/(1+e^-z)
Gradient:  ∂L/∂w = (1/n)Σ(ŷ-y)·x        ← same shape for both, only ŷ differs
```

**Production pattern:**
```python
from sklearn.linear_model import LinearRegression, LogisticRegression
model = LogisticRegression(class_weight='balanced')  # ALWAYS set this if imbalanced
model.fit(X_train, y_train)
```

**Edge cases:**
- Forgot to scale features → gradient descent converges slowly/unevenly
- `Ridge(alpha=0)` ≠ `LinearRegression()` — use LinearRegression explicitly for no regularization
- Threshold 0.5 is a DEFAULT, not a rule — tune it against your actual cost of FP vs FN

**How to make it better:**
- Feature scaling (StandardScaler) — always, no exceptions
- Add polynomial features if relationship is non-linear (but watch degree — see Module 5)
- Check residual plots for regression — pattern = missing something

---

## 2. Gradient Descent & Optimizers

**Math that matters:**
```
SGD:       w -= α·grad
Momentum:  v = β·v + (1-β)·grad;  w -= α·v                          (β=0.9)
Adam:      m = β₁·m+(1-β₁)·grad;  v = β₂·v+(1-β₂)·grad²             (β₁=0.9, β₂=0.999)
           m̂=m/(1-β₁ᵗ); v̂=v/(1-β₂ᵗ);  w -= α·m̂/(√v̂+ε)
```

**Production pattern:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)          # from scratch
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)           # fine-tuning pretrained
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
# Loop: zero_grad() → forward → loss → backward() → clip_grad_norm_() → step()
```

**Edge cases:**
- Forgot `optimizer.zero_grad()` → gradients ACCUMULATE across batches, silent explosion
- LR too high → loss diverges/NaN. Too low → painfully slow, looks "stuck"
- Fine-tuning with a full-training-scratch LR (1e-3) → destroys pretrained knowledge

**How to make it better:**
- Add `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)` — cheap insurance against explosion, esp. RNNs
- Use a scheduler — fixed LR the whole run is rarely optimal
- Warmup for transformers/fine-tuning (first ~10% of steps ramp LR up gradually)

---

## 3. Regularization

**Math that matters:**
```
Ridge (L2):  Loss = MSE + λΣw²      → shrinks weights, never exactly 0
Lasso (L1):  Loss = MSE + λΣ|w|     → can zero out weights (feature selection built in)
```

**Production pattern:**
```python
from sklearn.linear_model import Ridge, Lasso, LassoCV
model = Ridge(alpha=1.0)      # alpha = λ, tune via GridSearchCV/RandomizedSearchCV
lasso_cv = LassoCV(cv=5)      # auto-finds best alpha
```

**Edge cases:**
- λ=0 → no regularization, same as plain regression. λ→∞ → all weights → 0, severe underfit
- L1's feature selection is a side effect of the geometry (diamond-shaped constraint hits an axis) — not magic

**How to make it better:** if overfitting → increase λ. If underfitting → decrease λ. Always tune via CV, never guess.

---

## 4. Bias-Variance & Cross-Validation

**The single most useful diagnostic table in this entire cheat sheet:**
```
Train HIGH + Test HIGH   → HIGH BIAS      → bigger model, less regularization, more features
Train LOW  + Test HIGH   → HIGH VARIANCE  → more data, more regularization, simpler model, early stopping
Train LOW  + Test LOW    → balanced       → focus on features/data quality now
```

**Production pattern:**
```python
cv_result = cross_validate(model, X_train, y_train, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
gap = -cv_result['test_score'].mean() - (-cv_result['train_score'].mean())
# gap large → variance problem. train error itself high → bias problem.
```

**Edge cases:**
- Nested CV: NEVER report the inner CV score as final performance — it's optimistically biased (tuned on that exact data)
- `neg_mean_squared_error` — always flip the sign back: real_mse = -cv_scores
- StratifiedKFold for classification ALWAYS, even if "looks" balanced

**How to make it better:** if high variance → get more data (most reliable fix) before reaching for anything fancier.

---

## 5. Feature Engineering — Leak-Proof Pipelines

**Production pattern (THE pattern — memorize this shape):**
```python
preprocessor = ColumnTransformer([
    ('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                      ('scale', StandardScaler())]), numeric_cols),
    ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                      ('encode', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
])
full_pipeline = Pipeline([('prep', preprocessor), ('model', RandomForestClassifier())])
cross_val_score(full_pipeline, X_train, y_train, cv=5)   # leak-free, fits inside each fold
```

**Edge cases:**
- Fitting scaler on FULL data before split = leakage. Always split first.
- Target encoding without out-of-fold = leakage (worse than obvious — it's subtle)
- `OneHotEncoder(handle_unknown='ignore')` — ALWAYS in production, unseen category → all zeros not a crash
- `drop='first'` for linear models (avoid multicollinearity), skip it for tree models

**How to make it better:** log-transform right-skewed numeric features (price, income, counts) BEFORE scaling.

---

## 6. Trees, Random Forest, XGBoost

**Math that matters:**
```
Gini = 1 - Σpₖ²                    (0=pure, 0.5=max mixed, binary)
Bootstrap: sample n with replacement → ~63.2% unique, ~36.8% OOB (free validation)
XGBoost gain: uses gradient AND hessian (2nd order) — more principled than Gini
```

**Production pattern:**
```python
rf = RandomForestClassifier(n_estimators=200, max_features='sqrt', oob_score=True, class_weight='balanced')
xgb_model = xgb.XGBClassifier(n_estimators=1000, max_depth=4, learning_rate=0.05,
                              early_stopping_rounds=30, subsample=0.8, colsample_bytree=0.8)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
```

**Edge cases:**
- RF: deep trees ARE the point (bagging handles overfitting) — don't limit depth like a single tree
- XGBoost: SHALLOW trees (depth 3-6), opposite of RF — boosting needs weak learners
- XGBoost CAN overfit as n_estimators grows unlike RF — always use early_stopping_rounds
- `feature_importances_` (weight) is biased toward continuous features — use SHAP for real interpretation

**How to make it better:** Random Forest first as baseline (fast, few knobs). XGBoost if you need more accuracy and have time to tune.

---

## 7. Clustering & PCA

**Math that matters:**
```
K-Means objective: minimize Σ||x-centroid||²  (WCSS/inertia — always decreases with more K, can't use alone)
Silhouette: (b-a)/max(a,b)   -1 to +1, higher=better separation
PCA: eigenvector of covariance matrix with largest eigenvalue = direction of max variance
```

**Production pattern:**
```python
km = KMeans(n_clusters=k, init='k-means++', n_init=10)   # always k-means++, always n_init=10+
pca = PCA(n_components=0.95)                               # keep 95% variance, don't hardcode a number
```

**Edge cases:**
- ALWAYS scale before K-Means/PCA — unscaled = distance dominated by whatever has biggest raw numbers
- Silhouette score to pick K, not just the elbow (elbow is often ambiguous)
- DBSCAN's `eps` is hard to tune — no perfect formula, needs experimentation

**How to make it better:** profile each cluster (mean of each feature per cluster) to give clusters real business meaning, not just numbers.

---

## 8. Evaluation Metrics

**The decision table:**
```
Accuracy   → balanced classes only, otherwise LIES to you
Precision  → false alarms costly (spam, ad targeting)
Recall     → misses costly (cancer, fraud, default)
F1         → imbalanced + care about both (harmonic mean — punishes the WORSE score hard)
ROC-AUC    → general comparison, threshold-independent
PR-AUC     → severe imbalance — ROC-AUC lies when negatives vastly outnumber positives
```

**Production pattern:**
```python
class_weight='balanced'                          # try first, always
SMOTE(sampling_strategy=1.0)                      # only on TRAINING data, inside imblearn.Pipeline
precision_recall_curve(y_true, y_proba)           # find YOUR optimal threshold, 0.5 is not sacred
```

**Edge cases:**
- SMOTE on the whole dataset before CV split = leakage — must be inside the CV loop
- MSE/RMSE penalize large errors heavily (squared) — MAE doesn't, more robust to outliers
- Never impute or scale using TEST set statistics — training stats only, always

---
---

# PART 2 — DEEP LEARNING

---

## 9. CNNs

**Math that matters:**
```
Output size: H_out = (H - k + 2P)/S + 1
padding=1, kernel=3 → output size UNCHANGED (this exact combo is everywhere for a reason)
Flatten size = channels × height × width  (trace this by hand before writing Linear layer)
```

**Production pattern:**
```python
nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)   # size-preserving conv
nn.BatchNorm2d(out_ch)  → nn.ReLU()  → nn.MaxPool2d(2,2)   # standard block order
# ALWAYS: model.train() before training loop, model.eval() before inference
# ALWAYS: torch.no_grad() during inference
```

**Edge cases:**
- Wrong Linear layer input size = runtime crash. Trace shape through EVERY layer by hand or `print(x.shape)`.
- Augment TRAIN transform only — augmenting val/test gives an unreliable metric
- Transform-leak gotcha: build TWO ImageFolder instances (different transforms) + split by index, don't split one and expect different transforms
- Forgot `.convert('RGB')` on inference input → real uploads can be any mode (grayscale/RGBA)

**How to make it better:** checkpoint the BEST val model, not the last epoch. Small datasets overfit fast — epoch 20 often worse than epoch 8.

---

## 10. RNN / LSTM

**Math that matters:**
```
Forget gate: how much OLD memory to keep (σ, 0-1)
Input gate + candidate: what NEW info to add
Cell state: c = forget⊙c_old + input⊙candidate    ← addition/mult only, NOT matrix mult repeated
                                                       (this is WHY LSTMs dodge vanishing gradients)
hidden.shape = (num_layers × num_directions, batch, hidden_dim)
```

**Production pattern:**
```python
nn.Embedding(vocab_size, embed_dim, padding_idx=0)     # pad token always maps to zero, never trained
nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)  # forward + backward final states
pad_sequence(batch, batch_first=True, padding_value=0)    # dynamic per-batch padding
```

**Edge cases:**
- No custom `collate_fn` → DataLoader CRASHES on variable-length sequences. #1 beginner bug moving from toy to real text.
- Build vocab from TRAINING tokens only — same leakage rule as feature engineering
- `dict.get(word, unk_idx)` — handles out-of-vocab words gracefully instead of crashing
- Gradient clipping matters MORE here than CNNs — recurrent connections are exactly what explodes

**How to make it better:** bidirectional almost always helps if you have the full sequence upfront (not streaming/generation).

---

## 11. Transformers / Attention

**Math that matters (the whole architecture in one formula):**
```
Attention(Q,K,V) = softmax(QKᵀ/√d_k) × V
Q asks, K advertises, V delivers.
Multi-head = run this h times in parallel with different weights, concat, project.
Causal mask (GPT) = lower triangular — position i sees only 0..i, never future.
```

**Production pattern:**
```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
# fine-tune LR: 2e-5 (NOT 1e-3 — that's for training from scratch)
trainer = Trainer(model, args, train_dataset=tokenized)
trainer.train()   # same 5-step loop under the hood: zero_grad→forward→loss→backward→step
```

**Edge cases:**
- Tokenizer and model MUST come from the SAME model name — mixing silently produces garbage
- Attention cost is O(seq_len²) — doubling max_length QUADRUPLES compute
- Encoder (BERT) = bidirectional, understanding tasks. Decoder (GPT) = causal, generation tasks.

**How to make it better:** try `pipeline()` (zero training) before building your own fine-tuning pipeline — often good enough.

---
---

# PART 3 — GENAI / RAG ENGINEERING (the part most jobs actually want right now)

---

## 12. Embeddings & Semantic Search

**Math that matters:**
```
cosine_sim = (a·b)/(|a||b|)     for NORMALIZED vectors: cosine_sim = a·b  (just a dot product, fast)
Range: -1 to 1, practically 0 to 1 for text. >0.8 very similar. <0.3 unrelated.
```

**Production pattern:**
```python
model = SentenceTransformer('all-MiniLM-L6-v2')          # fast default, English
# 'paraphrase-multilingual-MiniLM-L12-v2'                  → for Hindi/mixed-language
embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
```

**Edge cases:**
- ALWAYS `normalize_embeddings=True` — makes cosine similarity a simple, fast dot product downstream
- General models fail on domain terms (GSTIN, "neelami", legal jargon) — measure this explicitly before assuming it's fine

**How to make it better:** if domain gap is real (measured, not assumed), fine-tune (LoRA/contrastive) before switching to a bigger general model.

---

## 13. FAISS & ChromaDB

**Production pattern:**
```python
# FAISS — fast, you manage metadata yourself
index = faiss.IndexFlatIP(dimension)                       # exact search, <100k vectors
index.add(embeddings.astype(np.float32))                    # ⭐ float32 required, silent bug if skipped
distances, indices = index.search(query_embeddings, k)

# ChromaDB — full vector DB, metadata + filtering built in
collection.add(ids=ids, embeddings=embeds, documents=texts, metadatas=metas)
collection.query(query_embeddings=[q], n_results=5, where={"chunk_type": {"$eq": "totals"}})
```

**Edge cases:**
- FAISS: metadata desyncs from vectors if you're not careful — keep them in lockstep always
- ChromaDB distance→similarity: `similarity = 1 - distance` for cosine space — easy to get backwards
- `upsert` not `add` if documents might get re-indexed — avoids crash on duplicate IDs

**How to make it better:** IndexIVFFlat or HNSW once corpus > 100k vectors — exact search doesn't scale past that.

---

## 14. Chunking Strategies — the single highest-leverage RAG decision

**The three strategies:**
```
Fixed-size    → naive baseline, ignores structure entirely, WILL split a rule from its exception
Sentence      → better, still doesn't know a table row isn't a sentence
Structure-aware → understands YOUR document's sections, each section = one chunk, never split mid-rule
```

**Production pattern:**
```python
# Structure-aware: detect section headers via regex, group lines, ONE chunk per section
# ⭐ ALWAYS explicitly verify: does the termination/default section still contain
# BOTH the rule AND its exception after chunking? Check the actual text, don't assume.
has_rule = "15 days" in chunk.text
has_exception = "7 days" in chunk.text  # both True = chunking preserved the full rule
```

**Edge cases:**
- A rule split from its exception is a SILENT, dangerous bug — no error thrown, just a wrong answer later
- Test multiple strategies, measure, don't assume structure-aware always wins on every document type

**How to make it better:** measure retrieval quality (Part 15 below) across strategies on YOUR real documents before committing to one.

---

## 15. RAG Evaluation

**Math that matters:**
```
Reciprocal Rank = 1/rank_of_first_correct_result     (found at #1→1.0, #2→0.5, not found→0)
MRR = average Reciprocal Rank across all test queries
Recall@K = correct_found_in_top_K / total_correct_that_exist
NDCG@K = like Recall@K but rewards finding it EARLIER (rank 1 > rank 5 even if both "in top 5")
```

**Production pattern:**
```python
def reciprocal_rank(relevant, retrieved):
    for rank, item in enumerate(retrieved, 1):
        if item in relevant: return 1.0/rank
    return 0.0
# Build 20-50 hand-verified (query, correct_chunk_id) pairs. Run every chunking/model
# combo through the SAME eval set. Compare numbers, don't guess.
```

**Edge cases:**
- "Looked right on 3 examples" is not evaluation — build a real query set with verified ground truth
- Break results down by query TYPE (factual/numeric/hindi/cross_section) — one overall number hides where it actually fails

**How to make it better:** bi-encoder (fast) retrieves top-20 → cross-encoder (slow, accurate) reranks to top-5. Never run cross-encoder on the full corpus.

---

## 16. Local LLMs / Ollama

**Math that matters:**
```
softmax(logits/temperature)
temperature→0 = deterministic (highest logit dominates). temperature↑ = more random.
Quantization: 7B params × 4 bytes (fp32) = 28GB. × 0.5 bytes (4-bit) ≈ 4GB.
```

**Production pattern:**
```python
requests.post(f"{OLLAMA_URL}/api/chat", json={
    "model": "mistral", "messages": messages, "stream": False,
    "options": {"temperature": 0.0, "num_predict": 400}       # 0.0-0.1 for extraction/factual
})
# ⭐ CONFIDENCE GATE — check BEFORE calling the LLM:
if best_retrieval_score < 0.3:
    return "not found"   # cheaper AND safer than letting the LLM guess from weak context
```

**Edge cases:**
- `response.raise_for_status()` ALWAYS — a silent failed request (Ollama not running) is a confusing bug otherwise
- System prompt = your output format CONTRACT. "Return ONLY valid JSON, no markdown" is what actually makes extraction reliable.
- Wrap JSON parsing in try/except with regex fallback — models occasionally produce near-valid JSON

**How to make it better:** `base_url` swap (Ollama ↔ OpenAI) lets you develop free locally, upgrade to paid API later — same code either way.

---

## 17. LoRA / QLoRA Fine-Tuning

**Math that matters:**
```
ΔW ≈ B×A     where A:(r,d_in), B:(d_out,r), r<<d_in,d_out
768×768 full = 589,824 params.  r=16 LoRA = 24,576 params (24x fewer).
Original W is FROZEN. B starts at zero → model = base model at step 0.
```

**Production pattern:**
```python
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"],
                         lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # should show <1% trainable — if ~100%, misconfigured

# Embedding fine-tuning uses TRIPLETS, not next-token prediction:
InputExample(texts=[anchor, positive, negative])
losses.MultipleNegativesRankingLoss(model)   # free negatives from other positives in same batch
```

**Edge cases:**
- `print_trainable_parameters()` near 100% = LoRA isn't actually applied, you're doing full fine-tuning by accident
- Bigger batch size = more free in-batch negatives = better training signal for MultipleNegativesRankingLoss

**How to make it better:** start r=16, alpha=32. Increase r only if the model clearly underperforms, don't guess-inflate it upfront.

---
---

# PART 4 — SHIPPING IT

---

## 18. Databases & Indexing

**Math that matters:**
```
No index: O(n) — check every row
With index: O(log n) — B-tree jump, ~17 checks for 100k rows vs up to 100,000
```

**Production pattern:**
```python
conn.execute("PRAGMA journal_mode=WAL")      # readers/writers don't block each other
conn.execute("PRAGMA foreign_keys=ON")       # OFF by default — CASCADE does nothing without this
conn.execute("INSERT INTO t VALUES (?,?,?)", (a,b,c))   # ALWAYS parameterized, never f-string SQL
```

**Edge cases:**
- `PRAGMA foreign_keys=ON` forgotten → CASCADE/RESTRICT rules silently do nothing, no error thrown
- f-string SQL = SQL injection vulnerability, not a style choice
- `EXPLAIN QUERY PLAN` to verify your index is actually being used ("SEARCH...USING INDEX" not "SCAN")

**How to make it better:** index every column you filter/join on frequently. Don't over-index — every index slows INSERT/UPDATE slightly.

---

## 19. Redis Caching

**Production pattern:**
```python
try:
    self.redis.ping(); self.available = True
except redis.ConnectionError:
    self.available = False    # app still works, just slower — NEVER let cache failure crash the app
self.redis.setex(key, ttl_seconds, json.dumps(value))    # atomic set+expiry
```

**Edge cases:** cache is ALWAYS optional — every get/set wrapped in try/except, fall back to computing fresh.

---

## 20. FastAPI

**Production pattern:**
```python
class Request(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)   # auto-validated, 422 if violated
@lru_cache(maxsize=1)
def get_repo(): return LoanRepository(...)    # created ONCE, reused every request
@app.post("/ask")
async def ask(req: Request, repo=Depends(get_repo)):
    background_tasks.add_task(slow_function, ...)   # respond NOW, work continues after
```

**Edge cases:** `async`/`await` = don't idly block during I/O, not "parallel execution." CORS middleware needed or your frontend can't call the API at all.

---

## 21. Docker

**Production pattern:**
```dockerfile
COPY requirements.txt .
RUN pip install -r requirements.txt   # ← cached if requirements.txt unchanged
COPY . .                               # ← code, changes every time — order matters for cache
USER appuser                           # never run as root
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Edge cases:** wrong COPY order = every code change invalidates the pip-install cache = 5min rebuilds instead of 10sec. `--host 0.0.0.0` required or app is unreachable from outside the container.

---
---

# THE ONE-PAGE SUMMARY (if you remember nothing else)

```
1. Bias-variance table: train/test gap tells you EXACTLY what to fix.
2. Leak-proof pipelines: fit on train, apply everywhere else, always.
3. Attention = Q asks, K advertises, V delivers.
4. Chunking > embedding model choice — a split rule is a silent dangerous bug.
5. LoRA freezes the original weights, trains two small matrices instead.
6. Always verify retrieval confidence BEFORE calling the LLM.
7. Index what you filter on. Parameterize every query. Enable foreign_keys.
8. Never trust AI-generated (or your own) output without running it and
   checking the number actually makes sense. This is the real skill.
```
