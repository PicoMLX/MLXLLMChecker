<!-- Mode: Manual + Reference -->

# MLXLLMChecker

Estimate the Unified Memory required to run an MLX model on Apple Silicon -- without downloading the weights.

Give it a HuggingFace model ID. It fetches `config.json` and file metadata (a few KB), calculates memory for weights, KV cache, and runtime overhead, and hands back a breakdown in bytes. The model weights stay on the server.

## What you'll do

- Add the package to your Swift project.
- Create an `MLXLLMChecker` instance.
- Call `estimateMemory(for:)` with a model ID.
- Read the result: total memory, per-component breakdown, and whether `mlx-swift-lm` supports the architecture.

## Before you start

- **macOS 15+** or **iOS 18+**.
- **Swift 6.2** toolchain (Xcode 26 or a recent trunk snapshot).
- Network access to `huggingface.co` (or a mirror).

## Install the package

Add MLXLLMChecker to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/picoMLX/MLXLLMChecker", branch: "main"),
]
```

Then add it to your target:

```swift
.target(
    name: "YourApp",
    dependencies: ["MLXLLMChecker"]
)
```

## Estimate memory for a single model

```swift
import MLXLLMChecker

let checker = MLXLLMChecker()
let estimate = try await checker.estimateMemory(for: "mlx-community/Qwen3-4B-4bit")

print(estimate.totalMemoryFormatted)   // e.g. "3.4 GB"
print(estimate.weightsMemory)          // bytes
print(estimate.kvCacheMemory)          // bytes
print(estimate.overheadMemory)         // bytes
print(estimate.isModelTypeSupported)   // true if mlx-swift-lm can run it
```

**Verify it worked**

- `totalMemoryFormatted` prints a human-readable string like `"3.4 GB"`.
- `weightSource` is `.safetensorsFileSize` when file sizes were available, `.calculated` when the library fell back to parameter estimation.

> **Try it now**
>
> Pick any model from `mlx-community` on HuggingFace and swap in the ID. Compare the estimate against the model card's stated size.

## Estimate memory for multiple models

The batch method fetches all models concurrently and streams results as they complete. The fastest model arrives first. Each result carries its `modelID`, so you always know which model it belongs to.

```swift
let checker = MLXLLMChecker()
let stream = checker.estimateMemory(for: [
    "mlx-community/Qwen3-4B-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
])

for await result in stream {
    switch result {
    case .success(let estimate):
        print("\(estimate.modelID): \(estimate.totalMemoryFormatted)")
    case .failure(let error):
        print("Error: \(error.localizedDescription)")
    }
}
```

The stream emits exactly one `Result<MemoryEstimate, Error>` per model ID, then finishes. Individual failures are `.failure` cases -- they do not terminate the stream or cancel other fetches.

Cancelling the consuming task cancels all in-flight fetches:

```swift
let task = Task {
    for await result in stream {
        // process...
    }
}
// Later:
task.cancel()  // cancels remaining network requests
```

## Context length resolution

When you omit `contextLength`, the library reads `max_position_embeddings` from the model's `config.json`. If that field is also absent, it falls back to 4096.

```swift
// Uses max_position_embeddings from config (e.g. 40960 for Qwen3-4B)
let estimate = try await checker.estimateMemory(for: "mlx-community/Qwen3-4B-4bit")
print(estimate.contextLength)  // 40960 (from config)
```

To override the config value, pass `contextLength` explicitly:

```swift
let estimate = try await checker.estimateMemory(
    for: "mlx-community/Qwen3-4B-4bit",
    contextLength: 32768
)
```

KV cache memory scales linearly with context length. Doubling it roughly doubles the KV cache component. The weights and overhead components stay the same.

Resolution order:

1. Explicit `contextLength` argument (if provided).
2. `max_position_embeddings` from `config.json` (if present).
3. `4096` (hardcoded fallback).

## Use KV cache quantization

Pass `kvQuantizationBits` to model a quantized KV cache (4-bit or 8-bit). This reduces the KV cache memory estimate.

```swift
let estimate = try await checker.estimateMemory(
    for: "mlx-community/Qwen3-4B-4bit",
    contextLength: 4096,
    kvQuantizationBits: 4
)
```

The library adds 10% overhead to the raw bit calculation to account for quantization metadata (scales and zero-points).

## Connect to a HuggingFace mirror

Users behind the Great Firewall (or anyone preferring a mirror) can point the checker at an alternative host:

```swift
let checker = MLXLLMChecker(
    mirrorURL: URL(string: "https://hf-mirror.com")!
)
let estimate = try await checker.estimateMemory(for: "mlx-community/Qwen3-4B-4bit")
```

To authenticate with a private mirror, pass a bearer token:

```swift
let checker = MLXLLMChecker(
    mirrorURL: URL(string: "https://hf-mirror.com")!,
    bearerToken: "hf_your_token_here"
)
```

For full control, inject your own `HubClient`:

```swift
import HuggingFace

let client = HubClient(host: myURL, bearerToken: myToken)
let checker = MLXLLMChecker(client: client)
```

## Check model type support

Every `MemoryEstimate` includes an `isModelTypeSupported` flag. This tells you whether `mlx-swift-lm` has a registered implementation for the model's `model_type` string.

```swift
if estimate.isModelTypeSupported {
    print("\(estimate.modelType) is supported -- safe to load.")
} else {
    print("\(estimate.modelType) is not supported by mlx-swift-lm.")
}
```

You can also query the registry directly:

```swift
SupportedModelTypes.isSupported("llama")    // true
SupportedModelTypes.isSupported("rwkv")     // false
SupportedModelTypes.llm.count               // 49 LLM architectures
SupportedModelTypes.vlm.count               // 13 VLM architectures
```

The supported set is static and mirrors `LLMTypeRegistry` and `VLMTypeRegistry` from `mlx-swift-lm`. Update `SupportedModelTypes.swift` when you bump the dependency.

---

# How memory is calculated

The total estimate is the sum of three components:

```
total = weights + kv_cache + overhead
```

Each component uses a primary method with a fallback when data is unavailable.

## 1. Weights memory

**Primary method: safetensors file sizes.**

The library calls `HubClient.listFiles()` to get the repository file tree. Every `.safetensors` file entry includes a `size` field (bytes). The library sums these values.

This is the exact on-disk weight footprint. No estimation needed.

**Fallback: parameter-based calculation.**

Triggered when file sizes are unavailable (the `listFiles` call fails, or no `.safetensors` files have size metadata). The library estimates the parameter count from the architecture config, then multiplies by effective bits per parameter.

Step 1 -- estimate parameters:

```
embedding       = vocab_size * hidden_size
attention/layer = hidden * heads * head_dim * 2           (Q + O projections)
                + hidden * kv_heads * head_dim * 2        (K + V projections)
ffn/layer       = 3 * hidden_size * intermediate_size     (gate + up + down)
norms/layer     = 2 * hidden_size
lm_head         = hidden_size * vocab_size                (0 if weights are tied)

total_params    = embedding + layers * (attention + ffn + norms) + lm_head + hidden_size
```

For **Mixture of Experts** models, the FFN term changes:

```
ffn/layer = 3 * hidden_size * moe_intermediate_size * num_experts
```

Step 2 -- compute effective bits per parameter:

```
effective_bits = quant_bits * (1 + 16 / (group_size * quant_bits))
```

The `16 / (group_size * quant_bits)` term accounts for the FP16 scale and zero-point stored per quantization group. A 4-bit model with group size 64 uses 4.25 effective bits per parameter.

If no quantization config is present, the library assumes FP16 (16 bits).

Step 3 -- multiply:

```
weight_bytes = total_params * effective_bits / 8
```

**How to tell which method was used:** check `estimate.weightSource`. It is `.safetensorsFileSize` or `.calculated`.

## 2. KV cache memory

The KV cache stores key and value tensors for every token in the context window, across every layer.

```
bytes_per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
kv_cache_bytes  = bytes_per_token * context_length
```

| Variable | Default | Source |
|----------|---------|--------|
| `layers` | -- | `num_hidden_layers` from config |
| `kv_heads` | `num_attention_heads` | `num_key_value_heads` if present (GQA) |
| `head_dim` | `hidden_size / num_attention_heads` | `head_dim` if present |
| `dtype_bytes` | 2.0 (FP16) | `kvQuantizationBits / 8 * 1.1` if KV quantization specified |

The factor of 2 accounts for both the key cache and the value cache.

**Fallback for missing `num_key_value_heads`:** the library uses `num_attention_heads` (standard multi-head attention, no GQA savings).

**Fallback for missing `head_dim`:** computed as `hidden_size / num_attention_heads`.

**Special case -- DeepSeek V3:** when `kv_lora_rank` and `qk_rope_head_dim` are both present, the effective cache dimension per head becomes `kv_lora_rank + qk_rope_head_dim` instead of `head_dim`. This models DeepSeek's compressed KV cache.

## 3. Runtime overhead

A fixed estimate for tokenizer data, activation buffers, and workspace memory.

```
tokenizer_overhead  = 100 MB
activation_overhead = 500 MB * min(hidden_size / 2048, 4.0)
total_overhead      = tokenizer_overhead + activation_overhead
```

| Hidden size | Scale factor | Total overhead |
|-------------|-------------|----------------|
| 768         | 0.375       | 288 MB         |
| 2048        | 1.0         | 600 MB         |
| 4096        | 2.0         | 1.1 GB         |
| 8192+       | 4.0 (cap)   | 2.1 GB         |

The cap at 4.0x prevents runaway estimates for very large hidden dimensions.

## Summary of fallbacks

| Component | Primary | Fallback | Trigger |
|-----------|---------|----------|---------|
| Context length | Explicit `contextLength` argument | `max_position_embeddings` from config, then 4096 | Caller passes `nil` |
| Weights | Sum `.safetensors` file sizes | Parameter count * effective bits | File listing fails or returns no sizes |
| KV heads | `num_key_value_heads` | `num_attention_heads` | Field absent from config |
| Head dim | `head_dim` | `hidden_size / num_attention_heads` | Field absent from config |
| Cache dim (DeepSeek) | `kv_lora_rank + qk_rope_head_dim` | Standard `head_dim` | Either field absent |
| KV dtype | FP16 (2 bytes) | `kvQuantizationBits / 8 * 1.1` | User passes `kvQuantizationBits` |
| Quantization bits | `quantization.bits` from config | 16.0 (FP16) | No quantization block in config |

---

# API reference

<!-- Mode: Reference -->

## `MLXLLMChecker`

The main entry point. A thread-safe, immutable object. Create one and reuse it.

### Initializers

| Initializer | Description |
|-------------|-------------|
| `init()` | Default HuggingFace endpoint (`huggingface.co`). |
| `init(mirrorURL: URL, bearerToken: String? = nil)` | Custom mirror host with optional authentication. |
| `init(client: HubClient)` | Pre-configured `HubClient` from the `HuggingFace` package. |

### Methods

#### `estimateMemory(for:contextLength:kvQuantizationBits:)` -- single model

```swift
public func estimateMemory(
    for modelID: String,
    contextLength: Int? = nil,
    kvQuantizationBits: Int? = nil
) async throws -> MemoryEstimate
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `modelID` | `String` | -- | HuggingFace model ID (`"namespace/model-name"`). |
| `contextLength` | `Int?` | `nil` | Token count for KV cache sizing. `nil` = use `max_position_embeddings` from config, then 4096. |
| `kvQuantizationBits` | `Int?` | `nil` | KV cache quantization (e.g. `4`, `8`). `nil` = FP16. |

**Returns:** `MemoryEstimate`.

**Throws:** `MLXLLMCheckerError`.

#### `estimateMemory(for:contextLength:kvQuantizationBits:)` -- batch (AsyncStream)

```swift
public func estimateMemory(
    for modelIDs: [String],
    contextLength: Int? = nil,
    kvQuantizationBits: Int? = nil
) -> AsyncStream<Result<MemoryEstimate, Error>>
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `modelIDs` | `[String]` | -- | Array of HuggingFace model IDs. |
| `contextLength` | `Int?` | `nil` | Token count for KV cache sizing. `nil` = each model uses its own `max_position_embeddings`, then 4096. |
| `kvQuantizationBits` | `Int?` | `nil` | KV cache quantization. `nil` = FP16. |

**Returns:** `AsyncStream<Result<MemoryEstimate, Error>>`. Results arrive in completion order (fastest first). Emits one element per model ID, then finishes. Individual failures are `.failure` cases -- they do not terminate the stream. Cancelling the consuming task cancels all in-flight fetches.

---

## `MemoryEstimate`

The result of a memory estimation. All memory values are in bytes.

| Property | Type | Description |
|----------|------|-------------|
| `modelID` | `String` | The model ID that was queried. |
| `modelType` | `String` | The `model_type` string from `config.json`. |
| `isModelTypeSupported` | `Bool` | `true` if `mlx-swift-lm` has a registered implementation. |
| `weightsMemory` | `UInt64` | Weight memory in bytes. |
| `kvCacheMemory` | `UInt64` | KV cache memory in bytes for the given context length. |
| `overheadMemory` | `UInt64` | Runtime overhead in bytes (tokenizer + activations). |
| `totalMemory` | `UInt64` | Computed. Sum of weights + KV cache + overhead. |
| `totalMemoryFormatted` | `String` | Human-readable total (e.g. `"5.2 GB"`). |
| `kvCacheDetails` | `KVCacheEstimate` | Detailed KV cache breakdown. |
| `weightSource` | `WeightSource` | `.safetensorsFileSize` or `.calculated`. |
| `contextLength` | `Int` | The context length used for this estimate. |
| `isMoE` | `Bool` | `true` if the model uses Mixture of Experts. |
| `quantization` | `QuantizationSummary?` | Quantization details, or `nil` for unquantized models. |

### `MemoryEstimate.formatBytes(_:)`

```swift
public static func formatBytes(_ bytes: UInt64) -> String
```

Formats any byte count as `"X.Y GB"` (>= 1 GB) or `"X MB"` (< 1 GB).

---

## `KVCacheEstimate`

| Property | Type | Description |
|----------|------|-------------|
| `layers` | `Int` | Number of transformer layers. |
| `kvHeads` | `Int` | Number of KV heads (after GQA resolution). |
| `headDim` | `Int` | Dimension per head (or compressed dim for DeepSeek). |
| `contextLength` | `Int` | Token count used. |
| `bytesPerToken` | `UInt64` | KV cache bytes consumed per token. |
| `totalBytes` | `UInt64` | `bytesPerToken * contextLength`. |
| `kvQuantizationBits` | `Int?` | The KV quantization used, or `nil` for FP16. |

---

## `QuantizationSummary`

| Property | Type | Description |
|----------|------|-------------|
| `bits` | `Int` | Quantization bits (e.g. `4`, `8`). |
| `groupSize` | `Int` | Quantization group size (e.g. `64`, `128`). |
| `effectiveBitsPerParam` | `Double` | Computed. Includes scale/zero-point overhead. |

---

## `WeightSource`

| Case | Meaning |
|------|---------|
| `.safetensorsFileSize` | Weight size summed from actual file metadata. Most accurate. |
| `.calculated` | Weight size estimated from parameter count and quantization config. |

---

## `SupportedModelTypes`

Static registry. No instantiation needed.

| Member | Type | Description |
|--------|------|-------------|
| `llm` | `Set<String>` | LLM `model_type` strings (49 entries). |
| `vlm` | `Set<String>` | VLM `model_type` strings (13 entries). |
| `all` | `Set<String>` | Union of `llm` and `vlm`. |
| `isSupported(_:)` | `(String) -> Bool` | Returns `true` if the type is in `all`. |

Source of truth: `LLMModelFactory.swift` and `VLMModelFactory.swift` in the `mlx-swift-lm` package. Update `SupportedModelTypes.swift` when bumping the dependency.

---

## `MLXLLMCheckerError`

| Case | Description |
|------|-------------|
| `.invalidModelID(String)` | Model ID is not in `"namespace/name"` format. |
| `.configNotFound(String)` | No `config.json` in the repository. |
| `.configParsingFailed(String, underlying: Error)` | `config.json` exists but could not be decoded. |
| `.missingRequiredField(String, model: String)` | A required field is absent from config. |
| `.networkError(String, underlying: Error)` | Network failure fetching from HuggingFace. |

All cases include the model ID for diagnostic context. Cases with `underlying` wrap the original error.

---

## `MemoryCalculator`

Pure computation. No network I/O. Useful for testing or building custom pipelines.

```swift
// Full estimate from a parsed config
let estimate = MemoryCalculator.estimate(
    config: myConfig,
    modelID: "my-org/my-model",
    safetensorsBytes: 2_300_000_000,  // or nil for fallback
    contextLength: 4096,
    kvQuantizationBits: nil
)

// Individual components
let (weightBytes, source) = MemoryCalculator.weightsEstimate(config: myConfig, safetensorsBytes: nil)
let kvEstimate = MemoryCalculator.kvCacheEstimate(config: myConfig, contextLength: 4096, kvQuantizationBits: nil)
let overhead = MemoryCalculator.overheadEstimate(config: myConfig)
```

---

## `ModelConfig`

Codable struct parsed from `config.json`. Required fields will throw `DecodingError` if absent.

**Required fields:**

| Property | JSON key | Type |
|----------|----------|------|
| `modelType` | `model_type` | `String` |
| `hiddenSize` | `hidden_size` | `Int` |
| `hiddenLayers` | `num_hidden_layers` | `Int` |
| `intermediateSize` | `intermediate_size` | `Int` |
| `attentionHeads` | `num_attention_heads` | `Int` |
| `vocabularySize` | `vocab_size` | `Int` |

**Optional fields:**

| Property | JSON key | Type | Used for |
|----------|----------|------|----------|
| `kvHeads` | `num_key_value_heads` | `Int?` | GQA KV cache sizing |
| `headDim` | `head_dim` | `Int?` | Override head dimension |
| `maxPositionEmbeddings` | `max_position_embeddings` | `Int?` | Default context length when caller omits `contextLength` |
| `tieWordEmbeddings` | `tie_word_embeddings` | `Bool?` | LM head param count |
| `quantization` | `quantization` | `QuantizationConfig?` | Weight size fallback |
| `numExperts` | `num_experts` | `Int?` | MoE detection |
| `numExpertsPerTok` | `num_experts_per_tok` | `Int?` | Active expert count |
| `moeIntermediateSize` | `moe_intermediate_size` | `Int?` | MoE FFN size |
| `kvLoraRank` | `kv_lora_rank` | `Int?` | DeepSeek compressed KV |
| `qkRopeHeadDim` | `qk_rope_head_dim` | `Int?` | DeepSeek compressed KV |

**Computed properties:**

| Property | Returns |
|----------|---------|
| `resolvedHeadDim` | `headDim ?? (hiddenSize / attentionHeads)` |
| `resolvedKVHeads` | `kvHeads ?? attentionHeads` |
| `isMoE` | `true` if `numExperts > 1` |
| `totalExperts` | `numExperts ?? 1` |

---

# Troubleshooting

- **`invalidModelID` error for a model that exists on HuggingFace:** the ID must contain exactly one `/`. `"Qwen3-4B-4bit"` fails; `"mlx-community/Qwen3-4B-4bit"` works. **Fix:** use the full `namespace/model-name` form.

- **`networkError` on every call:** check your network connection. If you are behind a firewall or in China, use `MLXLLMChecker(mirrorURL:)` with `https://hf-mirror.com`. **Fix:** verify connectivity with `curl https://huggingface.co/api/models`.

- **`configParsingFailed` for a model you know is valid:** the model's `config.json` may use field names this library does not parse, or it may be missing required fields (`hidden_size`, `num_hidden_layers`, etc.). **Fix:** inspect the model's `config.json` on HuggingFace and check it contains all six required fields.

- **`weightSource` is `.calculated` instead of `.safetensorsFileSize`:** the file listing API did not return size metadata, or the repository has no `.safetensors` files (GGUF-only models, for example). The calculated estimate is reasonable but less precise. **Fix:** no action needed; this is expected for some repositories.

- **`isModelTypeSupported` is `false` for a model you can load:** the `SupportedModelTypes` set may be out of date. **Fix:** update `SupportedModelTypes.swift` to match the current `LLMModelFactory.swift` and `VLMModelFactory.swift` in your `mlx-swift-lm` checkout.

- **Memory estimate seems too low:** the estimate covers weights, KV cache, and a fixed overhead. It does not account for speculative decoding buffers, large batch sizes, or vision encoder activations for VLMs. Treat the number as a minimum.

- **Memory estimate seems too high:** at short context lengths the overhead component dominates for small models. The 100 MB tokenizer + activation budget is conservative. Real usage may be lower.

---

# Architecture

```
MLXLLMChecker (public API)
    │
    ├── HubModelFetcher        fetches config.json + file listing via HubClient
    │       │
    │       └── HubClient      from swift-huggingface (injected)
    │
    ├── MemoryCalculator       pure math, no I/O
    │       │
    │       └── SupportedModelTypes    static model_type registry
    │
    └── ModelConfig            Codable config.json parser (Foundation only)
```

All types are `Sendable`. `MLXLLMChecker` is a `final class` with immutable stored properties. Batch estimation uses structured concurrency (`withTaskGroup`). No actors, no locks.

---

# Run the tests

```
swift test
```

Look for `Test run with 24 tests in 2 suites passed`. The test suites are:

- **ModelConfig JSON Parsing** -- decodes inline JSON fixtures for Llama, Qwen3, MoE, DeepSeek, minimal configs, and missing-field error cases.
- **MemoryCalculator** -- verifies weight estimation, parameter counting, effective bits, KV cache math, overhead scaling, formatting, and supported-type lookups.

All tests are offline. No network calls.
