import Foundation

/// Minimal Codable representation of a HuggingFace model's config.json.
///
/// Covers the superset of fields needed for memory estimation across
/// standard transformer, GQA, MoE, and DeepSeek architectures.
public struct ModelConfig: Sendable, Equatable {

    // MARK: - Required fields

    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let vocabularySize: Int

    // MARK: - Optional fields

    public let kvHeads: Int?
    public let headDim: Int?
    public let maxPositionEmbeddings: Int?
    public let tieWordEmbeddings: Bool?

    // MARK: - Quantization

    public let quantization: QuantizationConfig?

    // MARK: - MoE fields

    public let numExperts: Int?
    public let numExpertsPerTok: Int?
    public let moeIntermediateSize: Int?

    // MARK: - DeepSeek fields

    public let kvLoraRank: Int?
    public let qkRopeHeadDim: Int?

    // MARK: - Computed helpers

    /// Effective head dimension: explicit `head_dim` or `hidden_size / num_attention_heads`.
    public var resolvedHeadDim: Int {
        headDim ?? (hiddenSize / attentionHeads)
    }

    /// Effective number of KV heads: explicit `num_key_value_heads` or `num_attention_heads`.
    public var resolvedKVHeads: Int {
        kvHeads ?? attentionHeads
    }

    /// Whether this model uses Mixture of Experts.
    public var isMoE: Bool {
        numExperts != nil && (numExperts ?? 0) > 1
    }

    /// Total number of experts (1 for non-MoE models).
    public var totalExperts: Int {
        numExperts ?? 1
    }
}

// MARK: - QuantizationConfig

public struct QuantizationConfig: Sendable, Equatable, Codable {
    public let bits: Int
    public let groupSize: Int

    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
    }
}

// MARK: - Codable

extension ModelConfig: Codable {

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case quantization
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case kvLoraRank = "kv_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // Required fields
        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)

        // Optional fields
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads)
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
        quantization = try container.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
        numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts)
        numExpertsPerTok = try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)
        moeIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        kvLoraRank = try container.decodeIfPresent(Int.self, forKey: .kvLoraRank)
        qkRopeHeadDim = try container.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim)
    }
}
