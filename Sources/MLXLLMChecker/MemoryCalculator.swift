import Foundation

/// Pure computation for memory estimation. No I/O.
public struct MemoryCalculator: Sendable {

    // MARK: - Constants

    private static let tokenizerOverhead: UInt64 = 100 * 1024 * 1024       // 100 MB
    private static let baseActivationOverhead: UInt64 = 500 * 1024 * 1024  // 500 MB

    // MARK: - Public API

    /// Estimate total memory for a model given its config and weight file sizes.
    ///
    /// - Parameters:
    ///   - config: Parsed model configuration.
    ///   - modelID: HuggingFace model ID string.
    ///   - safetensorsBytes: Total bytes of .safetensors files (nil to use fallback calculation).
    ///   - totalDownloadBytes: Total bytes of all files in the repository (nil if unavailable).
    ///   - contextLength: Number of tokens for KV cache sizing.
    ///   - kvQuantizationBits: Optional KV cache quantization (e.g. 4 or 8).
    /// - Returns: A complete `MemoryEstimate`.
    public static func estimate(
        config: ModelConfig,
        modelID: String,
        safetensorsBytes: UInt64?,
        totalDownloadBytes: UInt64?,
        contextLength: Int,
        kvQuantizationBits: Int?
    ) -> MemoryEstimate {
        // Weights
        let (weightsMemory, weightSource) = weightsEstimate(
            config: config,
            safetensorsBytes: safetensorsBytes
        )

        // KV cache
        let kvEstimate = kvCacheEstimate(
            config: config,
            contextLength: contextLength,
            kvQuantizationBits: kvQuantizationBits
        )

        // Overhead
        let overhead = overheadEstimate(config: config)

        return MemoryEstimate(
            modelID: modelID,
            modelType: config.modelType,
            isModelTypeSupported: SupportedModelTypes.isSupported(config.modelType),
            weightsMemory: weightsMemory,
            kvCacheMemory: kvEstimate.totalBytes,
            overheadMemory: overhead,
            kvCacheDetails: kvEstimate,
            weightSource: weightSource,
            contextLength: contextLength,
            isMoE: config.isMoE,
            quantization: config.quantization.map {
                QuantizationSummary(bits: $0.bits, groupSize: $0.groupSize)
            },
            downloadSize: totalDownloadBytes
        )
    }

    // MARK: - Weights

    /// Determine weight memory from safetensors file sizes or parameter-based fallback.
    public static func weightsEstimate(
        config: ModelConfig,
        safetensorsBytes: UInt64?
    ) -> (bytes: UInt64, source: WeightSource) {
        if let bytes = safetensorsBytes, bytes > 0 {
            return (bytes, .safetensorsFileSize)
        }
        return (calculateWeightsFromParams(config: config), .calculated)
    }

    /// Fallback: estimate weight bytes from parameter count and quantization.
    static func calculateWeightsFromParams(config: ModelConfig) -> UInt64 {
        let params = estimateParameterCount(config: config)
        let bitsPerParam = effectiveBitsPerParam(config: config)
        return UInt64(Double(params) * bitsPerParam / 8.0)
    }

    /// Estimate total parameter count from architecture config.
    static func estimateParameterCount(config: ModelConfig) -> UInt64 {
        let hidden = UInt64(config.hiddenSize)
        let layers = UInt64(config.hiddenLayers)
        let heads = UInt64(config.attentionHeads)
        let headDim = UInt64(config.resolvedHeadDim)
        let vocab = UInt64(config.vocabularySize)

        // Embedding
        let embedding = vocab * hidden

        // Per-layer attention: Q, K, V projections + output projection
        let kvH = UInt64(config.resolvedKVHeads)
        let qProj = hidden * heads * headDim
        let kProj = hidden * kvH * headDim
        let vProj = hidden * kvH * headDim
        let oProj = heads * headDim * hidden
        let attention = qProj + kProj + vProj + oProj

        // Per-layer FFN
        let ffnSize: UInt64
        if config.isMoE {
            let moeIntermediate = UInt64(config.moeIntermediateSize ?? config.intermediateSize)
            let expertParams = 3 * hidden * moeIntermediate  // gate + up + down
            ffnSize = expertParams * UInt64(config.totalExperts)
        } else {
            ffnSize = 3 * hidden * UInt64(config.intermediateSize)  // gate + up + down
        }

        // Per-layer norms (2 per layer: attention + FFN)
        let norms = 2 * hidden

        let perLayer = attention + ffnSize + norms
        let allLayers = layers * perLayer

        // LM head
        let lmHead: UInt64
        if config.tieWordEmbeddings == true {
            lmHead = 0
        } else {
            lmHead = hidden * vocab
        }

        // Final norm
        let finalNorm = hidden

        return embedding + allLayers + lmHead + finalNorm
    }

    /// Effective bits per parameter, accounting for quantization scale overhead.
    static func effectiveBitsPerParam(config: ModelConfig) -> Double {
        guard let quant = config.quantization else {
            return 16.0  // fp16 default
        }
        let bits = Double(quant.bits)
        let groupSize = Double(quant.groupSize)
        let scaleOverhead = 16.0 / (groupSize * bits)
        return bits * (1.0 + scaleOverhead)
    }

    // MARK: - KV Cache

    /// Estimate KV cache memory for a given context length.
    public static func kvCacheEstimate(
        config: ModelConfig,
        contextLength: Int,
        kvQuantizationBits: Int?
    ) -> KVCacheEstimate {
        let layers = config.hiddenLayers
        let kvH = config.resolvedKVHeads

        // Effective cache dimension per head
        let cacheDim: Int
        if let loraRank = config.kvLoraRank, let ropeDim = config.qkRopeHeadDim {
            // DeepSeek V3: compressed KV cache
            cacheDim = loraRank + ropeDim
        } else {
            cacheDim = config.resolvedHeadDim
        }

        // Bytes per element
        let dtypeBytes: Double
        if let kvBits = kvQuantizationBits {
            dtypeBytes = Double(kvBits) / 8.0 * 1.1  // 10% overhead for quant metadata
        } else {
            dtypeBytes = 2.0  // fp16
        }

        // 2 × (K + V) × layers × kv_heads × head_dim × dtype_bytes
        let bytesPerToken = UInt64(2.0 * Double(layers) * Double(kvH) * Double(cacheDim) * dtypeBytes)
        let totalBytes = bytesPerToken * UInt64(contextLength)

        return KVCacheEstimate(
            layers: layers,
            kvHeads: kvH,
            headDim: cacheDim,
            contextLength: contextLength,
            bytesPerToken: bytesPerToken,
            totalBytes: totalBytes,
            kvQuantizationBits: kvQuantizationBits
        )
    }

    // MARK: - Overhead

    /// Estimate runtime overhead (tokenizer, activations, workspace buffers).
    public static func overheadEstimate(config: ModelConfig) -> UInt64 {
        let scale = min(Double(config.hiddenSize) / 2048.0, 4.0)
        let activations = UInt64(Double(baseActivationOverhead) * scale)
        return tokenizerOverhead + activations
    }
}
