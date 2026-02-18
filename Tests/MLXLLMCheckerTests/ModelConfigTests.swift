import Testing
import Foundation
@testable import MLXLLMChecker

@Suite("ModelConfig JSON Parsing")
struct ModelConfigTests {

    // MARK: - Llama-style config

    @Test("Parse standard Llama config")
    func parseLlama() throws {
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
        }
        """
        let config = try JSONDecoder().decode(ModelConfig.self, from: Data(json.utf8))

        #expect(config.modelType == "llama")
        #expect(config.hiddenSize == 4096)
        #expect(config.hiddenLayers == 32)
        #expect(config.intermediateSize == 11008)
        #expect(config.attentionHeads == 32)
        #expect(config.vocabularySize == 32000)
        #expect(config.kvHeads == 32)
        #expect(config.maxPositionEmbeddings == 4096)
        #expect(config.tieWordEmbeddings == false)
        #expect(config.quantization == nil)
        #expect(config.isMoE == false)
        #expect(config.resolvedHeadDim == 128)
        #expect(config.resolvedKVHeads == 32)
    }

    // MARK: - Qwen3 quantized config with GQA

    @Test("Parse Qwen3-4B quantized config with GQA")
    func parseQwen3Quantized() throws {
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
            "max_position_embeddings": 40960,
            "tie_word_embeddings": true,
            "quantization": {
                "bits": 4,
                "group_size": 64
            }
        }
        """
        let config = try JSONDecoder().decode(ModelConfig.self, from: Data(json.utf8))

        #expect(config.modelType == "qwen3")
        #expect(config.hiddenSize == 2560)
        #expect(config.hiddenLayers == 36)
        #expect(config.kvHeads == 8)
        #expect(config.headDim == 80)
        #expect(config.resolvedHeadDim == 80)
        #expect(config.resolvedKVHeads == 8)
        #expect(config.tieWordEmbeddings == true)
        #expect(config.quantization?.bits == 4)
        #expect(config.quantization?.groupSize == 64)
    }

    // MARK: - MoE config (Mixtral-style)

    @Test("Parse MoE config")
    func parseMoE() throws {
        let json = """
        {
            "model_type": "mistral",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "quantization": {
                "bits": 4,
                "group_size": 64
            }
        }
        """
        let config = try JSONDecoder().decode(ModelConfig.self, from: Data(json.utf8))

        #expect(config.isMoE == true)
        #expect(config.totalExperts == 8)
        #expect(config.numExpertsPerTok == 2)
    }

    // MARK: - DeepSeek V3 config

    @Test("Parse DeepSeek V3 config with compressed KV cache fields")
    func parseDeepSeekV3() throws {
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
        let config = try JSONDecoder().decode(ModelConfig.self, from: Data(json.utf8))

        #expect(config.modelType == "deepseek_v3")
        #expect(config.kvLoraRank == 512)
        #expect(config.qkRopeHeadDim == 64)
        #expect(config.isMoE == true)
        #expect(config.totalExperts == 256)
        #expect(config.numExpertsPerTok == 8)
        #expect(config.moeIntermediateSize == 2048)
    }

    // MARK: - Minimal config (only required fields)

    @Test("Parse minimal config with only required fields")
    func parseMinimal() throws {
        let json = """
        {
            "model_type": "custom",
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "vocab_size": 30522
        }
        """
        let config = try JSONDecoder().decode(ModelConfig.self, from: Data(json.utf8))

        #expect(config.modelType == "custom")
        #expect(config.kvHeads == nil)
        #expect(config.headDim == nil)
        #expect(config.quantization == nil)
        #expect(config.resolvedHeadDim == 64)  // 768 / 12
        #expect(config.resolvedKVHeads == 12)   // falls back to attention heads
        #expect(config.isMoE == false)
        #expect(config.totalExperts == 1)
    }

    // MARK: - Missing required field

    @Test("Fail on missing required field")
    func missingRequiredField() {
        let json = """
        {
            "model_type": "llama",
            "hidden_size": 4096
        }
        """
        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder().decode(ModelConfig.self, from: Data(json.utf8))
        }
    }
}
