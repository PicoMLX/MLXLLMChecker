import Testing
import Foundation
@testable import MLXLLMChecker

@Suite("MemoryCalculator")
struct MemoryCalculatorTests {

    // MARK: - Helpers

    /// Creates a basic Llama-7B-like config for testing.
    private func makeLlamaConfig(
        quantBits: Int? = nil,
        quantGroupSize: Int? = nil
    ) -> ModelConfig {
        let json = """
        {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": false
            \(quantBits != nil ? """
            , "quantization": {
                "bits": \(quantBits!),
                "group_size": \(quantGroupSize ?? 64)
            }
            """ : "")
        }
        """
        return try! JSONDecoder().decode(ModelConfig.self, from: Data(json.utf8))
    }

    /// Creates a Qwen3-4B-like config (GQA, quantized).
    private func makeQwen3Config() -> ModelConfig {
        let json = """
        {
            "model_type": "qwen3",
            "hidden_size": 2560,
            "num_hidden_layers": 36,
            "intermediate_size": 6912,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 80,
            "vocab_size": 151936,
            "tie_word_embeddings": true,
            "quantization": {
                "bits": 4,
                "group_size": 64
            }
        }
        """
        return try! JSONDecoder().decode(ModelConfig.self, from: Data(json.utf8))
    }

    /// Creates a DeepSeek V3 config with compressed KV cache.
    private func makeDeepSeekConfig() -> ModelConfig {
        let json = """
        {
            "model_type": "deepseek_v3",
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "intermediate_size": 18432,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "vocab_size": 129280,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048,
            "quantization": {
                "bits": 4,
                "group_size": 128
            }
        }
        """
        return try! JSONDecoder().decode(ModelConfig.self, from: Data(json.utf8))
    }

    // MARK: - Weight estimation with safetensors sizes

    @Test("Use safetensors file sizes when available")
    func weightFromSafetensors() {
        let config = makeLlamaConfig()
        let (bytes, source) = MemoryCalculator.weightsEstimate(
            config: config,
            safetensorsBytes: 3_500_000_000
        )
        #expect(bytes == 3_500_000_000)
        #expect(source == .safetensorsFileSize)
    }

    @Test("Fall back to calculated when safetensors bytes is nil")
    func weightFallbackNil() {
        let config = makeLlamaConfig()
        let (bytes, source) = MemoryCalculator.weightsEstimate(
            config: config,
            safetensorsBytes: nil
        )
        #expect(source == .calculated)
        #expect(bytes > 0)
    }

    @Test("Fall back to calculated when safetensors bytes is 0")
    func weightFallbackZero() {
        let config = makeLlamaConfig()
        let (_, source) = MemoryCalculator.weightsEstimate(
            config: config,
            safetensorsBytes: 0
        )
        #expect(source == .calculated)
    }

    // MARK: - Parameter count estimation

    @Test("Llama-7B parameter count in reasonable range")
    func llamaParamCount() {
        let config = makeLlamaConfig()
        let params = MemoryCalculator.estimateParameterCount(config: config)
        // Llama-7B has ~6.7B params
        let billions = Double(params) / 1e9
        #expect(billions > 6.0 && billions < 8.0)
    }

    // MARK: - Effective bits per param

    @Test("FP16 when no quantization")
    func fp16Bits() {
        let config = makeLlamaConfig()
        let bits = MemoryCalculator.effectiveBitsPerParam(config: config)
        #expect(bits == 16.0)
    }

    @Test("4-bit quantization effective bits include scale overhead")
    func quantized4Bits() {
        let config = makeLlamaConfig(quantBits: 4, quantGroupSize: 64)
        let bits = MemoryCalculator.effectiveBitsPerParam(config: config)
        // 4 * (1 + 16/(64*4)) = 4 * (1 + 0.0625) = 4.25
        #expect(abs(bits - 4.25) < 0.001)
    }

    // MARK: - KV cache

    @Test("KV cache for Llama-7B at 4096 context")
    func kvCacheLlama() {
        let config = makeLlamaConfig()
        let kv = MemoryCalculator.kvCacheEstimate(
            config: config,
            contextLength: 4096,
            kvQuantizationBits: nil
        )
        #expect(kv.layers == 32)
        #expect(kv.kvHeads == 32)
        #expect(kv.headDim == 128)
        #expect(kv.contextLength == 4096)
        // bytes_per_token = 2 * 32 * 32 * 128 * 2 = 524288
        #expect(kv.bytesPerToken == 524_288)
        // total = 524288 * 4096 = 2,147,483,648
        #expect(kv.totalBytes == 2_147_483_648)
    }

    @Test("KV cache with GQA (fewer KV heads)")
    func kvCacheGQA() {
        let config = makeQwen3Config()
        let kv = MemoryCalculator.kvCacheEstimate(
            config: config,
            contextLength: 4096,
            kvQuantizationBits: nil
        )
        #expect(kv.kvHeads == 8)
        #expect(kv.headDim == 80)
        // bytes_per_token = 2 * 36 * 8 * 80 * 2 = 92160
        #expect(kv.bytesPerToken == 92_160)
    }

    @Test("KV cache with quantization reduces memory")
    func kvCacheQuantized() {
        let config = makeLlamaConfig()
        let kvFP16 = MemoryCalculator.kvCacheEstimate(
            config: config,
            contextLength: 4096,
            kvQuantizationBits: nil
        )
        let kv4Bit = MemoryCalculator.kvCacheEstimate(
            config: config,
            contextLength: 4096,
            kvQuantizationBits: 4
        )
        #expect(kv4Bit.totalBytes < kvFP16.totalBytes)
        #expect(kv4Bit.kvQuantizationBits == 4)
    }

    @Test("DeepSeek V3 uses compressed KV cache dimension")
    func kvCacheDeepSeek() {
        let config = makeDeepSeekConfig()
        let kv = MemoryCalculator.kvCacheEstimate(
            config: config,
            contextLength: 4096,
            kvQuantizationBits: nil
        )
        // Should use kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576
        #expect(kv.headDim == 576)
    }

    // MARK: - Overhead

    @Test("Overhead scales with hidden size")
    func overheadScaling() {
        let small = makeLlamaConfig()  // hidden=4096
        let overheadSmall = MemoryCalculator.overheadEstimate(config: small)

        // Overhead should be tokenizer (100MB) + activations (500MB * scale)
        // For hidden=4096: scale = min(4096/2048, 4) = 2.0
        // overhead = 100MB + 500MB * 2.0 = 1100MB
        let expectedMB: UInt64 = 1100
        let tolerance: UInt64 = 10 * 1024 * 1024  // 10 MB
        #expect(abs(Int64(overheadSmall) - Int64(expectedMB * 1024 * 1024)) < Int64(tolerance))
    }

    // MARK: - Full estimation

    @Test("Full estimate with safetensors produces correct total")
    func fullEstimate() {
        let config = makeQwen3Config()
        let estimate = MemoryCalculator.estimate(
            config: config,
            modelID: "mlx-community/Qwen3-4B-4bit",
            safetensorsBytes: 2_300_000_000,
            totalDownloadBytes: 2_400_000_000,
            contextLength: 4096,
            kvQuantizationBits: nil
        )

        #expect(estimate.modelID == "mlx-community/Qwen3-4B-4bit")
        #expect(estimate.modelType == "qwen3")
        #expect(estimate.isModelTypeSupported == true)
        #expect(estimate.weightSource == .safetensorsFileSize)
        #expect(estimate.weightsMemory == 2_300_000_000)
        #expect(estimate.kvCacheMemory > 0)
        #expect(estimate.overheadMemory > 0)
        #expect(estimate.totalMemory == estimate.weightsMemory + estimate.kvCacheMemory + estimate.overheadMemory)
        #expect(estimate.isMoE == false)
        #expect(estimate.quantization?.bits == 4)
        #expect(estimate.downloadSize == 2_400_000_000)
        #expect(estimate.downloadSizeFormatted == "2.2 GB")
    }

    @Test("Download size is nil when file listing unavailable")
    func downloadSizeNil() {
        let config = makeQwen3Config()
        let estimate = MemoryCalculator.estimate(
            config: config,
            modelID: "mlx-community/Qwen3-4B-4bit",
            safetensorsBytes: 2_300_000_000,
            totalDownloadBytes: nil,
            contextLength: 4096,
            kvQuantizationBits: nil
        )
        #expect(estimate.downloadSize == nil)
        #expect(estimate.downloadSizeFormatted == nil)
    }

    @Test("Unsupported model type is flagged")
    func unsupportedModelType() {
        let json = """
        {
            "model_type": "unknown_arch",
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "vocab_size": 30522
        }
        """
        let config = try! JSONDecoder().decode(ModelConfig.self, from: Data(json.utf8))
        let estimate = MemoryCalculator.estimate(
            config: config,
            modelID: "test/unknown",
            safetensorsBytes: 500_000_000,
            totalDownloadBytes: 500_100_000,
            contextLength: 4096,
            kvQuantizationBits: nil
        )
        #expect(estimate.isModelTypeSupported == false)
    }

    // MARK: - Formatting

    @Test("Format bytes as GB")
    func formatGB() {
        let formatted = MemoryEstimate.formatBytes(5_368_709_120)  // 5.0 GB
        #expect(formatted == "5.0 GB")
    }

    @Test("Format bytes as MB")
    func formatMB() {
        let formatted = MemoryEstimate.formatBytes(524_288_000)  // ~500 MB
        #expect(formatted == "500 MB")
    }

    // MARK: - SupportedModelTypes

    @Test("Known LLM types are supported")
    func knownLLMTypes() {
        #expect(SupportedModelTypes.isSupported("llama"))
        #expect(SupportedModelTypes.isSupported("qwen3"))
        #expect(SupportedModelTypes.isSupported("deepseek_v3"))
        #expect(SupportedModelTypes.isSupported("mistral"))
    }

    @Test("Known VLM types are supported")
    func knownVLMTypes() {
        #expect(SupportedModelTypes.isSupported("qwen2_vl"))
        #expect(SupportedModelTypes.isSupported("pixtral"))
        #expect(SupportedModelTypes.isSupported("gemma3"))
    }

    @Test("Unknown type is not supported")
    func unknownType() {
        #expect(!SupportedModelTypes.isSupported("unknown_model"))
        #expect(!SupportedModelTypes.isSupported(""))
    }
}
