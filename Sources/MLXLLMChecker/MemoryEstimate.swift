import Foundation

/// How the weight memory size was determined.
public enum WeightSource: String, Sendable, Equatable {
    /// Summed from actual .safetensors file sizes in the repository.
    case safetensorsFileSize
    /// Calculated from model parameters and quantization config.
    case calculated
}

/// Summary of the model's quantization configuration.
public struct QuantizationSummary: Sendable, Equatable {
    public let bits: Int
    public let groupSize: Int

    /// Effective bits per parameter including scale/zero-point overhead.
    public var effectiveBitsPerParam: Double {
        let scaleOverhead = 16.0 / (Double(groupSize) * Double(bits))
        return Double(bits) * (1.0 + scaleOverhead)
    }
}

/// Details of the KV cache memory estimation.
public struct KVCacheEstimate: Sendable, Equatable {
    public let layers: Int
    public let kvHeads: Int
    public let headDim: Int
    public let contextLength: Int
    public let bytesPerToken: UInt64
    public let totalBytes: UInt64
    public let kvQuantizationBits: Int?
}

/// Complete memory estimation breakdown for a model.
public struct MemoryEstimate: Sendable, Equatable {
    public let modelID: String
    public let modelType: String
    public let isModelTypeSupported: Bool
    public let weightsMemory: UInt64
    public let kvCacheMemory: UInt64
    public let overheadMemory: UInt64
    public let kvCacheDetails: KVCacheEstimate
    public let weightSource: WeightSource
    public let contextLength: Int
    public let isMoE: Bool
    public let quantization: QuantizationSummary?

    /// Total download size for the repository in bytes, or `nil` if file listing was unavailable.
    public let downloadSize: UInt64?

    /// Total estimated memory: weights + KV cache + overhead.
    public var totalMemory: UInt64 {
        weightsMemory + kvCacheMemory + overheadMemory
    }

    /// Human-readable total memory string (e.g. "5.2 GB").
    public var totalMemoryFormatted: String {
        Self.formatBytes(totalMemory)
    }

    /// Human-readable download size string (e.g. "2.3 GB"), or `nil` if unavailable.
    public var downloadSizeFormatted: String? {
        downloadSize.map { Self.formatBytes($0) }
    }

    /// Human-readable formatting for any byte value.
    public static func formatBytes(_ bytes: UInt64) -> String {
        let gb = Double(bytes) / (1024.0 * 1024.0 * 1024.0)
        if gb >= 1.0 {
            return String(format: "%.1f GB", gb)
        }
        let mb = Double(bytes) / (1024.0 * 1024.0)
        return String(format: "%.0f MB", mb)
    }
}
