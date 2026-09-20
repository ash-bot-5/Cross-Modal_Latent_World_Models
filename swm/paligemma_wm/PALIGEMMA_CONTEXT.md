# PaliGemmaWM Schema

This package is a HuggingFace-compatible extension of the PaliGemma VLM that adds an **action projector** — a linear layer that injects robot action sequences into the language model's token embedding space alongside image and text.

---

## Files

### `configuration_paligemma_wm.py`

**Class: `PaliGemmaWMConfig(PretrainedConfig)`**

Stores all architectural hyperparameters. Registered under `model_type = "paligemma_wm"`.

| Parameter | Default | Description |
|---|---|---|
| `vision_config` | SigLIP (1152-dim, 27 layers, patch 14, image 224) | Vision encoder config |
| `text_config` | Gemma (2048-dim, 18 layers) | Language model config |
| `action_input_dim` | -1 (must be set) | Dimensionality of raw action vectors (e.g. 2 for LangTable, 5 for OGBench) |
| `projection_dim` | 2048 | Output dim of both multimodal projectors (must match LM hidden size) |
| `hidden_size` | 2048 | LM hidden size |
| `image_token_id` | 256000 | Token ID for `<image>` placeholder |
| `action_token_id` | 256001 | Token ID for `<action>` placeholder |
| `vocab_size` | 257152 | Total vocabulary size |

Key behaviors:
- `attribute_map` aliases `image_token_id ↔ image_token_index` and `action_token_id ↔ action_token_index` for backward compatibility.
- Always sets `text_config.use_bidirectional_attention = True` so the prefix (image + action + question) attends bidirectionally.
- `to_dict()` strips the deprecated `_ignore_index` key from serialized output.

---

### `modeling_paligemma_wm.py`

#### Output dataclasses

| Class | Extends | Extra fields |
|---|---|---|
| `PaligemmaWMModelOutputWithPast` | `BaseModelOutputWithPast` | `image_hidden_states`, `action_hidden_states` |
| `PaliGemmaWMCausalLMOutputWithPast` | `ModelOutput` | `loss`, `logits`, `past_key_values`, `hidden_states`, `attentions`, `image_hidden_states`, `action_hidden_states` |

#### Projector modules

**`PaliGemmaWMMultiModalProjector`**
- `nn.Linear(vision_config.hidden_size → vision_config.projection_dim, bias=True)`
- Projects SigLIP image patch features into the LM embedding space.
- Output is scaled by `1 / sqrt(text_config.hidden_size)`.

**`PaliGemmaWMMultiModalActionProjector`**
- `nn.Linear(action_input_dim → vision_config.projection_dim, bias=True)`
- Projects raw action vectors into the LM embedding space.
- Infinity-padded positions are filtered out before projection.

#### Attention masking

**`token_type_ids_mask_function`** — returns a per-element mask function that enables bidirectional attention within image token blocks (same image group). Used as an `or_mask_function` to undo causal masking for the prompt prefix.

**`create_causal_mask_mapping`** — wraps `create_masks_for_generate` to apply the bidirectional prefix mask. Inverts `token_type_ids` (0=prefix, 1=suffix in PaliGemmaWM convention, opposite of Gemma3) before delegating to the shared Gemma3 logic.

#### Model classes

**`PaliGemmaWMPreTrainedModel(PreTrainedModel)`**
- Base class. Supports Flash Attention 2, SDPA, and Flex Attention.
- `_no_split_modules` lists both projectors so they are never split across devices.

**`PaliGemmaWMModel(PaliGemmaWMPreTrainedModel)`**

The core multimodal backbone (no LM head).

Submodules:
- `vision_tower` — SigLIP encoder loaded via `AutoModel`
- `multi_modal_projector` — `PaliGemmaWMMultiModalProjector`
- `action_projector` — `PaliGemmaWMMultiModalActionProjector`
- `language_model` — Gemma backbone loaded via `AutoModel`

`forward(input_ids, pixel_values, action_values, ...)` pipeline:
1. Embed `input_ids` via the LM embedding table.
2. If `pixel_values` provided: encode with `vision_tower` → project → scatter into `<image>` placeholder positions.
3. If `action_values` provided: filter `inf`-padded entries → project with `action_projector` → scatter into `<action>` placeholder positions.
4. Build causal mask via `create_causal_mask_mapping` (bidirectional for prefix tokens).
5. Run `language_model` and return `PaligemmaWMModelOutputWithPast`.

Key methods:
- `get_image_features(pixel_values)` → `(num_images, image_seq_len, embed_dim)`
- `get_action_features(action_values)` → `(batch * seq_len_filtered, embed_dim)`
- `get_placeholder_image_mask(...)` / `get_placeholder_action_mask(...)` — validates token count matches feature count, returns boolean scatter mask.

**`PaliGemmaWMForConditionalGeneration(PaliGemmaWMPreTrainedModel, GenerationMixin)`**

Wraps `PaliGemmaWMModel` and adds an `lm_head` (`nn.Linear(hidden_size → vocab_size, bias=False)`).

`forward(...)` → `PaliGemmaWMCausalLMOutputWithPast`:
- Delegates to `PaliGemmaWMModel`, then applies `lm_head`.
- If `labels` provided: computes cross-entropy loss with `ignore_index=-100`, masking padding tokens. Uses a shifted (next-token) formulation; attention mask is applied before flattening.
- `logits_to_keep` (int or tensor): slices logits to avoid computing the full vocab projection when only the last token is needed (used at inference).

`prepare_inputs_for_generation(...)`:
- Passes `pixel_values` and `action_values` only on the first generation step (`cache_position[0] == 0`).
- Adds 1 to `position_ids` (PaliGemma uses 1-indexed positions).

`create_masks_for_generate(...)` (static): delegates to `create_causal_mask_mapping`.

Checkpoint conversion mapping (for loading legacy checkpoints saved before the model/head split):
```
language_model.model.* → model.language_model.*
language_model.lm_head.* → lm_head.*
vision_tower.*           → model.vision_tower.*
multi_modal_projector.*  → model.multi_modal_projector.*
action_projector.*       → model.action_projector.*
```

---

### `processing_paligemma_wm.py`

**Class: `PaliGemmaWMProcessor(ProcessorMixin)`**

Combines a `SiglipImageProcessor` and a `GemmaTokenizerFast` into one processor. Adds two special tokens on construction:
- `<image>` (token ID stored as `self.image_token_id`)
- `<action>` (token ID stored as `self.action_token_id`)

Also adds 1024 `<loc####>` and 128 `<seg###>` extra tokens (inherited from PaliGemma).

**`build_string_from_input`** (module-level helper) — constructs the full prompt string:
```
<image> × image_seq_length  <action> × num_actions  <bos>  {question text}\n
```

**`__call__(images, actions, text, ...)`** → `BatchFeature`:

Inputs:
- `images`: PIL / np / tensor, single or batched
- `actions`: `List[Tensor]` of shape `(num_actions, action_dim)` per batch item, or a single `(B, num_actions, action_dim)` tensor; `None` if no actions
- `text`: question string(s)
- `suffix` (kwarg): answer string(s) for training; appended after the prompt with `token_type_ids=1`

Processing steps:
1. `_process_actions(actions)` → `(padded_action_tensor | None, list_of_lengths)`. Pads a list of variable-length tensors with `torch.inf`; a batch tensor is returned as-is.
2. Builds `input_strings` via `build_string_from_input` (one per batch item).
3. Tokenizes via `GemmaTokenizerFast`; always requests `token_type_ids`.
4. Processes images via `SiglipImageProcessor` → `pixel_values`.
5. If `suffix` provided: appends EOS, computes `labels` where prefix positions are masked to -100.

Output keys: `input_ids`, `attention_mask`, `token_type_ids`, `pixel_values`, `action_values` (if actions given), `labels` (if suffix given).

**`_process_actions(inputs)`**:
- `np.ndarray` → converted to `torch.Tensor` (shape `(B, num_actions, action_dim)`)
- `torch.Tensor` (3D) → returned as-is with lengths = `[num_actions] * B`
- `List[Tensor]` → `pad_sequence(..., padding_value=inf)` + lengths list
- All-zero-length list → returns `(None, lengths)`

---

### `__init__.py`# PaliGemmaWM Schema

This package is a HuggingFace-compatible extension of the PaliGemma VLM that adds an **action projector** — a linear layer that injects robot action sequences into the language model's token embedding space alongside image and text.

---

## Files

### `configuration_paligemma_wm.py`

**Class: `PaliGemmaWMConfig(PretrainedConfig)`**

Stores all architectural hyperparameters. Registered under `model_type = "paligemma_wm"`.

| Parameter | Default | Description |
|---|---|---|
| `vision_config` | SigLIP (1152-dim, 27 layers, patch 14, image 224) | Vision encoder config |
| `text_config` | Gemma (2048-dim, 18 layers) | Language model config |
| `action_input_dim` | -1 (must be set) | Dimensionality of raw action vectors (e.g. 2 for LangTable, 5 for OGBench) |
| `projection_dim` | 2048 | Output dim of both multimodal projectors (must match LM hidden size) |
| `hidden_size` | 2048 | LM hidden size |
| `image_token_id` | 256000 | Token ID for `<image>` placeholder |
| `action_token_id` | 256001 | Token ID for `<action>` placeholder |
| `vocab_size` | 257152 | Total vocabulary size |

Key behaviors:
- `attribute_map` aliases `image_token_id ↔ image_token_index` and `action_token_id ↔ action_token_index` for backward compatibility.
- Always sets `text_config.use_bidirectional_attention = True` so the prefix (image + action + question) attends bidirectionally.
- `to_dict()` strips the deprecated `_ignore_index` key from serialized output.

---

### `modeling_paligemma_wm.py`

#### Output dataclasses

| Class | Extends | Extra fields |
|---|---|---|
| `PaligemmaWMModelOutputWithPast` | `BaseModelOutputWithPast` | `image_hidden_states`, `action_hidden_states` |
| `PaliGemmaWMCausalLMOutputWithPast` | `ModelOutput` | `loss`, `logits`, `past_key_values`, `hidden_states`, `attentions`, `image_hidden_states`, `action_hidden_states` |

#### Projector modules

**`PaliGemmaWMMultiModalProjector`**
- `nn.Linear(vision_config.hidden_size → vision_config.projection_dim, bias=True)`
- Projects SigLIP image patch features into the LM embedding space.
- Output is scaled by `1 / sqrt(text_config.hidden_size)`.

**`PaliGemmaWMMultiModalActionProjector`**
- `nn.Linear(action_input_dim → vision_config.projection_dim, bias=True)`
- Projects raw action vectors into the LM embedding space.
- Infinity-padded positions are filtered out before projection.

#### Attention masking

**`token_type_ids_mask_function`** — returns a per-element mask function that enables bidirectional attention within image token blocks (same image group). Used as an `or_mask_function` to undo causal masking for the prompt prefix.

**`create_causal_mask_mapping`** — wraps `create_masks_for_generate` to apply the bidirectional prefix mask. Inverts `token_type_ids` (0=prefix, 1=suffix in PaliGemmaWM convention, opposite of Gemma3) before delegating to the shared Gemma3 logic.

#### Model classes

**`PaliGemmaWMPreTrainedModel(PreTrainedModel)`**
- Base class. Supports Flash Attention 2, SDPA, and Flex Attention.
- `_no_split_modules` lists both projectors so they are never split across devices.

**`PaliGemmaWMModel(PaliGemmaWMPreTrainedModel)`**

The core multimodal backbone (no LM head).

Submodules:
- `vision_tower` — SigLIP encoder loaded via `AutoModel`
- `multi_modal_projector` — `PaliGemmaWMMultiModalProjector`
- `action_projector` — `PaliGemmaWMMultiModalActionProjector`
- `language_model` — Gemma backbone loaded via `AutoModel`

`forward(input_ids, pixel_values, action_values, ...)` pipeline:
1. Embed `input_ids` via the LM embedding table.
2. If `pixel_values` provided: encode with `vision_tower` → project → scatter into `<image>` placeholder positions.
3. If `action_values` provided: filter `inf`-padded entries → project with `action_projector` → scatter into `<action>` placeholder positions.
4. Build causal mask via `create_causal_mask_mapping` (bidirectional for prefix tokens).
5. Run `language_model` and return `PaligemmaWMModelOutputWithPast`.

Key methods:
- `get_image_features(pixel_values)` → `(num_images, image_seq_len, embed_dim)`
- `get_action_features(action_values)` → `(batch * seq_len_filtered, embed_dim)`
- `get_placeholder_image_mask(...)` / `get_placeholder_action_mask(...)` — validates token count matches feature count, returns boolean scatter mask.

**`PaliGemmaWMForConditionalGeneration(PaliGemmaWMPreTrainedModel, GenerationMixin)`**

Wraps `PaliGemmaWMModel` and adds an `lm_head` (`nn.Linear(hidden_size → vocab_size, bias=False)`).

`forward(...)` → `PaliGemmaWMCausalLMOutputWithPast`:
- Delegates to `PaliGemmaWMModel`, then applies `lm_head`.
- If `labels` provided: computes cross-entropy loss with `ignore_index=-100`, masking padding tokens. Uses a shifted (next-token) formulation; attention mask is applied before flattening.
- `logits_to_keep` (int or tensor): slices logits to avoid computing the full vocab projection when only the last token is needed (used at inference).

`prepare_inputs_for_generation(...)`:
- Passes `pixel_values` and `action_values` only on the first generation step (`cache_position[0] == 0`).
- Adds 1 to `position_ids` (PaliGemma uses 1-indexed positions).

`create_masks_for_generate(...)` (static): delegates to `create_causal_mask_mapping`.

Checkpoint conversion mapping (for loading legacy checkpoints saved before the model/head split):
```
language_model.model.* → model.language_model.*
language_model.lm_head.* → lm_head.*
vision_tower.*           → model.vision_tower.*
multi_modal_projector.*  → model.multi_modal_projector.*
action_projector.*       → model.action_projector.*
```

---

### `processing_paligemma_wm.py`

**Class: `PaliGemmaWMProcessor(ProcessorMixin)`**

Combines a `SiglipImageProcessor` and a `GemmaTokenizerFast` into one processor. Adds two special tokens on construction:
- `<image>` (token ID stored as `self.image_token_id`)
- `<action>` (token ID stored as `self.action_token_id`)

Also adds 1024 `<loc####>` and 128 `<seg###>` extra tokens (inherited from PaliGemma).

**`build_string_from_input`** (module-level helper) — constructs the full prompt string:
```
<image> × image_seq_length  <action> × num_actions  <bos>  {question text}\n
```

**`__call__(images, actions, text, ...)`** → `BatchFeature`:

Inputs:
- `images`: PIL / np / tensor, single or batched
- `actions`: `List[Tensor]` of shape `(num_actions, action_dim)` per batch item, or a single `(B, num_actions, action_dim)` tensor; `None` if no actions
- `text`: question string(s)
- `suffix` (kwarg): answer string(s) for training; appended after the prompt with `token_type_ids=1`

Processing steps:
1. `_process_actions(actions)` → `(padded_action_tensor | None, list_of_lengths)`. Pads a list of variable-length tensors with `torch.inf`; a batch tensor is returned as-is.
2. Builds `input_strings` via `build_string_from_input` (one per batch item).
3. Tokenizes via `GemmaTokenizerFast`; always requests `token_type_ids`.
4. Processes images via `SiglipImageProcessor` → `pixel_values`.
5. If `suffix` provided: appends EOS, computes `labels` where prefix positions are masked to -100.

Output keys: `input_ids`, `attention_mask`, `token_type_ids`, `pixel_values`, `action_values` (if actions given), `labels` (if suffix given).

**`_process_actions(inputs)`**:
- `np.ndarray` → converted to `torch.Tensor` (shape `(B, num_actions, action_dim)`)
- `torch.Tensor` (3D) → returned as-is with lengths = `[num_actions] * B`
- `List[Tensor]` → `pad_sequence(..., padding_value=inf)` + lengths list
- All-zero-length list → returns `(None, lengths)`

---

### `__init__.py`

Registers all four classes with the HuggingFace Auto API:

| Auto class | Registered class |
|---|---|
| `AutoConfig` | `PaliGemmaWMConfig` |
| `AutoModel` | `PaliGemmaWMModel` |
| `AutoModelForImageTextToText` | `PaliGemmaWMForConditionalGeneration` |
| `AutoProcessor` | `PaliGemmaWMProcessor` |

Also attempts to register the checkpoint weight-renaming map via `register_checkpoint_conversion_mapping` (gracefully skipped on stock HuggingFace transformers that don't expose this API).

---
# PaliGemmaWM Schema

This package is a HuggingFace-compatible extension of the PaliGemma VLM that adds an **action projector** — a linear layer that injects robot action sequences into the language model's token embedding space alongside image and text.

---

## Files

### `configuration_paligemma_wm.py`

**Class: `PaliGemmaWMConfig(PretrainedConfig)`**

Stores all architectural hyperparameters. Registered under `model_type = "paligemma_wm"`.

| Parameter | Default | Description |
|---|---|---|
| `vision_config` | SigLIP (1152-dim, 27 layers, patch 14, image 224) | Vision encoder config |
| `text_config` | Gemma (2048-dim, 18 layers) | Language model config |
| `action_input_dim` | -1 (must be set) | Dimensionality of raw action vectors (e.g. 2 for LangTable, 5 for OGBench) |
| `projection_dim` | 2048 | Output dim of both multimodal projectors (must match LM hidden size) |
| `hidden_size` | 2048 | LM hidden size |
| `image_token_id` | 256000 | Token ID for `<image>` placeholder |
| `action_token_id` | 256001 | Token ID for `<action>` placeholder |
| `vocab_size` | 257152 | Total vocabulary size |

Key behaviors:
- `attribute_map` aliases `image_token_id ↔ image_token_index` and `action_token_id ↔ action_token_index` for backward compatibility.
- Always sets `text_config.use_bidirectional_attention = True` so the prefix (image + action + question) attends bidirectionally.
- `to_dict()` strips the deprecated `_ignore_index` key from serialized output.

---

### `modeling_paligemma_wm.py`

#### Output dataclasses

| Class | Extends | Extra fields |
|---|---|---|
| `PaligemmaWMModelOutputWithPast` | `BaseModelOutputWithPast` | `image_hidden_states`, `action_hidden_states` |
| `PaliGemmaWMCausalLMOutputWithPast` | `ModelOutput` | `loss`, `logits`, `past_key_values`, `hidden_states`, `attentions`, `image_hidden_states`, `action_hidden_states` |

#### Projector modules

**`PaliGemmaWMMultiModalProjector`**
- `nn.Linear(vision_config.hidden_size → vision_config.projection_dim, bias=True)`
- Projects SigLIP image patch features into the LM embedding space.
- Output is scaled by `1 / sqrt(text_config.hidden_size)`.

**`PaliGemmaWMMultiModalActionProjector`**
- `nn.Linear(action_input_dim → vision_config.projection_dim, bias=True)`
- Projects raw action vectors into the LM embedding space.
- Infinity-padded positions are filtered out before projection.

#### Attention masking

**`token_type_ids_mask_function`** — returns a per-element mask function that enables bidirectional attention within image token blocks (same image group). Used as an `or_mask_function` to undo causal masking for the prompt prefix.

**`create_causal_mask_mapping`** — wraps `create_masks_for_generate` to apply the bidirectional prefix mask. Inverts `token_type_ids` (0=prefix, 1=suffix in PaliGemmaWM convention, opposite of Gemma3) before delegating to the shared Gemma3 logic.

#### Model classes

**`PaliGemmaWMPreTrainedModel(PreTrainedModel)`**
- Base class. Supports Flash Attention 2, SDPA, and Flex Attention.
- `_no_split_modules` lists both projectors so they are never split across devices.

**`PaliGemmaWMModel(PaliGemmaWMPreTrainedModel)`**

The core multimodal backbone (no LM head).

Submodules:
- `vision_tower` — SigLIP encoder loaded via `AutoModel`
- `multi_modal_projector` — `PaliGemmaWMMultiModalProjector`
- `action_projector` — `PaliGemmaWMMultiModalActionProjector`
- `language_model` — Gemma backbone loaded via `AutoModel`

`forward(input_ids, pixel_values, action_values, ...)` pipeline:
1. Embed `input_ids` via the LM embedding table.
2. If `pixel_values` provided: encode with `vision_tower` → project → scatter into `<image>` placeholder positions.
3. If `action_values` provided: filter `inf`-padded entries → project with `action_projector` → scatter into `<action>` placeholder positions.
4. Build causal mask via `create_causal_mask_mapping` (bidirectional for prefix tokens).
5. Run `language_model` and return `PaligemmaWMModelOutputWithPast`.

Key methods:
- `get_image_features(pixel_values)` → `(num_images, image_seq_len, embed_dim)`
- `get_action_features(action_values)` → `(batch * seq_len_filtered, embed_dim)`
- `get_placeholder_image_mask(...)` / `get_placeholder_action_mask(...)` — validates token count matches feature count, returns boolean scatter mask.

**`PaliGemmaWMForConditionalGeneration(PaliGemmaWMPreTrainedModel, GenerationMixin)`**

Wraps `PaliGemmaWMModel` and adds an `lm_head` (`nn.Linear(hidden_size → vocab_size, bias=False)`).

`forward(...)` → `PaliGemmaWMCausalLMOutputWithPast`:
- Delegates to `PaliGemmaWMModel`, then applies `lm_head`.
- If `labels` provided: computes cross-entropy loss with `ignore_index=-100`, masking padding tokens. Uses a shifted (next-token) formulation; attention mask is applied before flattening.
- `logits_to_keep` (int or tensor): slices logits to avoid computing the full vocab projection when only the last token is needed (used at inference).

`prepare_inputs_for_generation(...)`:
- Passes `pixel_values` and `action_values` only on the first generation step (`cache_position[0] == 0`).
- Adds 1 to `position_ids` (PaliGemma uses 1-indexed positions).

`create_masks_for_generate(...)` (static): delegates to `create_causal_mask_mapping`.

Checkpoint conversion mapping (for loading legacy checkpoints saved before the model/head split):
```
language_model.model.* → model.language_model.*
language_model.lm_head.* → lm_head.*
vision_tower.*           → model.vision_tower.*
multi_modal_projector.*  → model.multi_modal_projector.*
action_projector.*       → model.action_projector.*
```

---

### `processing_paligemma_wm.py`

**Class: `PaliGemmaWMProcessor(ProcessorMixin)`**

Combines a `SiglipImageProcessor` and a `GemmaTokenizerFast` into one processor. Adds two special tokens on construction:
- `<image>` (token ID stored as `self.image_token_id`)
- `<action>` (token ID stored as `self.action_token_id`)

Also adds 1024 `<loc####>` and 128 `<seg###>` extra tokens (inherited from PaliGemma).

**`build_string_from_input`** (module-level helper) — constructs the full prompt string:
```
<image> × image_seq_length  <action> × num_actions  <bos>  {question text}\n
```

**`__call__(images, actions, text, ...)`** → `BatchFeature`:

Inputs:
- `images`: PIL / np / tensor, single or batched
- `actions`: `List[Tensor]` of shape `(num_actions, action_dim)` per batch item, or a single `(B, num_actions, action_dim)` tensor; `None` if no actions
- `text`: question string(s)
- `suffix` (kwarg): answer string(s) for training; appended after the prompt with `token_type_ids=1`

Processing steps:
1. `_process_actions(actions)` → `(padded_action_tensor | None, list_of_lengths)`. Pads a list of variable-length tensors with `torch.inf`; a batch tensor is returned as-is.
2. Builds `input_strings` via `build_string_from_input` (one per batch item).
3. Tokenizes via `GemmaTokenizerFast`; always requests `token_type_ids`.
4. Processes images via `SiglipImageProcessor` → `pixel_values`.
5. If `suffix` provided: appends EOS, computes `labels` where prefix positions are masked to -100.

Output keys: `input_ids`, `attention_mask`, `token_type_ids`, `pixel_values`, `action_values` (if actions given), `labels` (if suffix given).

**`_process_actions(inputs)`**:
- `np.ndarray` → converted to `torch.Tensor` (shape `(B, num_actions, action_dim)`)
- `torch.Tensor` (3D) → returned as-is with lengths = `[num_actions] * B`
- `List[Tensor]` → `pad_sequence(..., padding_value=inf)` + lengths list
- All-zero-length list → returns `(None, lengths)`

---

### `__init__.py`

Registers all four classes with the HuggingFace Auto API:

| Auto class | Registered class |
|---|---|
| `AutoConfig` | `PaliGemmaWMConfig` |
| `AutoModel` | `PaliGemmaWMModel` |
| `AutoModelForImageTextToText` | `PaliGemmaWMForConditionalGeneration` |
| `AutoProcessor` | `PaliGemmaWMProcessor` |

Also attempts to register the checkpoint weight-renaming map via `register_checkpoint_conversion_mapping` (gracefully skipped on stock HuggingFace transformers that don't expose this API).

---

## Token sequence layout (at inference)

```
[ <image> × image_seq_length ]  [ <action> × num_actions ]  [ <bos> ]  [ question tokens ]  [ \n ]
```

- Image and action placeholder embeddings are overwritten in `PaliGemmaWMModel.forward` before the LM runs.
- `<action>` count equals the number of non-padding action steps (padding = `inf`, filtered out).
- Bidirectional attention spans the entire prefix (image + action + question + `\n`); causal masking applies only to any generated suffix.
- At inference, logits are read from the **last token position** of the prefix; softmax over `yes`/`no` token IDs gives the reward probability.

## Token sequence layout (at inference)

```
[ <image> × image_seq_length ]  [ <action> × num_actions ]  [ <bos> ]  [ question tokens ]  [ \n ]
```

- Image and action placeholder embeddings are overwritten in `PaliGemmaWMModel.forward` before the LM runs.
- `<action>` count equals the number of non-padding action steps (padding = `inf`, filtered out).
- Bidirectional attention spans the entire prefix (image + action + question + `\n`); causal masking applies only to any generated suffix.
- At inference, logits are read from the **last token position** of the prefix; softmax over `yes`/`no` token IDs gives the reward probability.


Registers all four classes with the HuggingFace Auto API:

| Auto class | Registered class |
|---|---|
| `AutoConfig` | `PaliGemmaWMConfig` |
| `AutoModel` | `PaliGemmaWMModel` |
| `AutoModelForImageTextToText` | `PaliGemmaWMForConditionalGeneration` |
| `AutoProcessor` | `PaliGemmaWMProcessor` |

Also attempts to register the checkpoint weight-renaming map via `register_checkpoint_conversion_mapping` (gracefully skipped on stock HuggingFace transformers that don't expose this API).

---

## Token sequence layout (at inference)

```
[ <image> × image_seq_length ]  [ <action> × num_actions ]  [ <bos> ]  [ question tokens ]  [ \n ]
```

- Image and action placeholder embeddings are overwritten in `PaliGemmaWMModel.forward` before the LM runs.
- `<action>` count equals the number of non-padding action steps (padding = `inf`, filtered out).
- Bidirectional attention spans the entire prefix (image + action + question + `\n`); causal masking applies only to any generated suffix.
- At inference, logits are read from the **last token position** of the prefix; softmax over `yes`/`no` token IDs gives the reward probability.
