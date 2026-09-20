# Plan: Multi-QA z_sem Caching (`num_qa_pairs` parameter)

## Context

Currently `cache_zsem.py` runs one Q&A pair per forward pass and extracts `hidden_states[-1][:, -1, :]` (the EOS-position hidden state) as z_sem. The goal is to concatenate N Q&A pairs into a single prompt per forward pass, so the resulting z_sem is a richer, multi-question goal latent. The planner can then condition on a single vector that encodes the full task description rather than a sparse single-question signal.

---

## Answers to your four questions

### Q1 — Prompt format and hidden-state extraction position

The processor call in `compute_zsem_batch` passes:
- `text=questions` — a single question string per batch item
- `suffix=answers` — `"yes"` or `"no"` per batch item

`build_string_from_input` constructs:
```
<image> × image_seq_length   <bos>   {question_text}\n
```
The suffix tokens (`yes`/`no` + EOS) are appended by the tokenizer as `token_type_ids=1`.

The full sequence seen by the model:
```
[image tokens] [BOS] [question text] [\n] [answer tokens] [EOS]
```

Hidden state is extracted at **`[:, -1, :]` = the EOS token position** (last token of the suffix). This is the final position after the answer, which attends bidirectionally to everything in the prefix (image + question) and causally to the suffix.

### Q2 — Does the processor need changes?

**No.** `build_string_from_input` just inserts whatever string you supply as `question_text`. Multi-QA concatenation is just string construction in the caller. The processor and tokenizer handle arbitrary text verbatim — no changes to `processing_paligemma_wm.py` are needed.

### Q3 — Sequence length constraints

There is **no explicit `max_length`** in `PaliGemmaWMProcessor` or `PaliGemmaWMConfig`. The underlying Gemma backbone has an architectural context window (typically 8192 tokens). For practical N:
- Image prefix: 256 tokens (fixed, 224×224 / patch-14)
- BOS + `\n`: 2 tokens
- Per Q&A pair: ~15–20 tokens (question + answer)
- N=4 pairs: ~256 + 2 + 80 ≈ 338 tokens — no issue

For N up to ~20, there is effectively no constraint worth guarding against at this scale.

### Q4 — Minimal set of files to edit

**Only `scripts/cache_zsem.py`.**  
The processor, model, and config are untouched.

---

## Plan

### File: `scripts/cache_zsem.py` only

#### Step 1 — Add config constant
Add `NUM_QA_PAIRS: int = 4` near the top of the Config block. This is how many Q&A pairs are concatenated per forward pass. Default 4 gives one z_sem per timestep per category for `task_relevant_qa` (which always has exactly 4 pairs).

#### Step 2 — Add `build_multi_qa_text` helper
A pure function that takes a list of `(question, answer)` pairs and returns `(text, suffix)` ready for the processor:

```python
def build_multi_qa_text(pairs):
    # pairs: List[Tuple[str, str]]  — (question, answer) for each of N Q&A pairs
    if len(pairs) == 1:
        return pairs[0][0], pairs[0][1]
    # Embed first N-1 pairs verbatim in the question string; last answer is suffix
    prefix_parts = [f"{q}\n{a}" for q, a in pairs[:-1]]
    last_q, last_a = pairs[-1]
    text = "\n".join(prefix_parts) + "\n" + last_q
    return text, last_a
```

This preserves the current extraction pattern: last answer goes in `suffix`, z_sem is at the EOS position.

#### Step 3 — Modify `process_episode`

Change item-building to group QA pairs into chunks of `NUM_QA_PAIRS` per `(category, t)`:

For each category at each timestep `t`, collect all available `(question, answer)` pairs and split into sequential groups of size `NUM_QA_PAIRS` (last group may be smaller). Each group becomes one item that will produce one z_sem.

New item structure:
```python
(category, t, group_idx, [(q0,a0), (q1,a1), ...])
```

Output array shapes change accordingly:
```
num_rel_groups  = ceil(4     / NUM_QA_PAIRS)
num_irr_groups  = ceil(n_irr / NUM_QA_PAIRS)
num_cmp_groups  = ceil(n_cmp / NUM_QA_PAIRS)

arr_rel   shape: (T, num_rel_groups, 2048)
arr_wrong shape: (T, num_rel_groups, 2048)
arr_irr   shape: (T, num_irr_groups, 2048)
arr_cmp   shape: (T, num_cmp_groups, 2048)
```

When `NUM_QA_PAIRS=1`, all shapes are identical to the current output. When `NUM_QA_PAIRS=4` for task_relevant categories, shape becomes `(T, 1, 2048)`.

#### Step 4 — Update the batch loop

In the `for start in range(0, len(items), BATCH_SIZE)` loop:
- For each group item, call `build_multi_qa_text(group_pairs)` to get `(text, suffix)`.
- Pass lists of texts and suffixes to `compute_zsem_batch` (no changes to that function needed; it already accepts arbitrary strings).
- Scatter the returned z_sem into `arr_*[t, group_idx]` instead of `arr_*[t, qi]`.

#### Step 5 — No changes needed to
- `swm/paligemma_wm/processing_paligemma_wm.py`
- `swm/paligemma_wm/modeling_paligemma_wm.py`
- `swm/paligemma_wm/configuration_paligemma_wm.py`

---

## Verification

After implementation, run on a small subset:
```bash
python scripts/cache_zsem.py
```
Check that:
1. The `.npz` file for one episode loads correctly and `arr['task_relevant_qa'].shape` is `(T, ceil(4/NUM_QA_PAIRS), 2048)`.
2. With `NUM_QA_PAIRS=1`, output shapes match the pre-existing cached files exactly.
3. With `NUM_QA_PAIRS=4`, `task_relevant_qa` has shape `(T, 1, 2048)` — confirmed by loading a `.npz` with `np.load`.
