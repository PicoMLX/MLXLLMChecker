/// Static registry of model_type strings supported by mlx-swift-lm.
///
/// These sets mirror the registrations in:
/// - `LLMTypeRegistry`: `.build/checkouts/mlx-swift-lm/Libraries/MLXLLM/LLMModelFactory.swift`
/// - `VLMTypeRegistry`: `.build/checkouts/mlx-swift-lm/Libraries/MLXVLM/VLMModelFactory.swift`
///
/// Update these sets when the mlx-swift-lm dependency is updated.
public struct SupportedModelTypes: Sendable {

    /// LLM model types from LLMTypeRegistry.
    public static let llm: Set<String> = [
        "mistral", "llama", "phi", "phi3", "phimoe", "gemma", "gemma2",
        "gemma3", "gemma3_text", "gemma3n", "qwen2", "qwen3", "qwen3_moe",
        "qwen3_next", "minicpm", "starcoder2", "cohere", "openelm",
        "internlm2", "deepseek_v3", "granite", "granitemoehybrid",
        "mimo", "mimo_v2_flash", "minimax", "glm4", "glm4_moe",
        "glm4_moe_lite", "acereason", "falcon_h1", "bitnet", "smollm3",
        "ernie4_5", "lfm2", "baichuan_m1", "exaone4", "gpt_oss",
        "lille-130m", "olmoe", "olmo2", "olmo3", "bailing_moe",
        "lfm2_moe", "nanochat", "nemotron_h", "afmoe", "jamba_3b",
        "mistral3", "apertus",
    ]

    /// VLM model types from VLMTypeRegistry.
    public static let vlm: Set<String> = [
        "paligemma", "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "idefics3",
        "gemma3", "smolvlm", "fastvlm", "llava_qwen2", "pixtral",
        "mistral3", "lfm2_vl", "lfm2-vl",
    ]

    /// All supported model types (union of LLM + VLM).
    public static let all: Set<String> = llm.union(vlm)

    /// Check if a model_type string is supported by mlx-swift-lm.
    public static func isSupported(_ modelType: String) -> Bool {
        all.contains(modelType)
    }
}
