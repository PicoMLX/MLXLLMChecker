import Foundation
import HuggingFace

/// Estimates the Unified Memory required to run MLX models on Apple Silicon
/// without downloading model weights.
///
/// Given a HuggingFace model ID, fetches only `config.json` and file metadata,
/// then calculates memory for weights, KV cache, and runtime overhead.
public final class MLXLLMChecker: Sendable {

    private let fetcher: HubModelFetcher

    /// Create a checker using the default HuggingFace endpoint.
    public init() {
        self.fetcher = HubModelFetcher(client: HubClient())
    }

    /// Create a checker using a mirror URL (e.g. `https://hf-mirror.com` for users in China).
    ///
    /// - Parameters:
    ///   - mirrorURL: Base URL of the HuggingFace mirror.
    ///   - bearerToken: Optional authentication token.
    public init(mirrorURL: URL, bearerToken: String? = nil) {
        self.fetcher = HubModelFetcher(
            client: HubClient(host: mirrorURL, bearerToken: bearerToken)
        )
    }

    /// Create a checker using a pre-configured `HubClient`.
    ///
    /// - Parameter client: A configured HuggingFace Hub client.
    public init(client: HubClient) {
        self.fetcher = HubModelFetcher(client: client)
    }

    // MARK: - Single Model

    /// Estimate memory required to run a single model.
    ///
    /// - Parameters:
    ///   - modelID: HuggingFace model ID (e.g. "mlx-community/Qwen3-4B-4bit").
    ///   - contextLength: Number of tokens for KV cache estimation. When `nil`,
    ///     uses `max_position_embeddings` from the model's config, falling back to 4096.
    ///   - kvQuantizationBits: Optional KV cache quantization bits (e.g. 4 or 8).
    /// - Returns: A `MemoryEstimate` with full breakdown.
    public func estimateMemory(
        for modelID: String,
        contextLength: Int? = nil,
        kvQuantizationBits: Int? = nil
    ) async throws -> MemoryEstimate {
        let result = try await fetcher.fetch(modelID: modelID)
        let resolvedContextLength = contextLength
            ?? result.config.maxPositionEmbeddings
            ?? 4096

        return MemoryCalculator.estimate(
            config: result.config,
            modelID: modelID,
            safetensorsBytes: result.safetensorsBytes,
            contextLength: resolvedContextLength,
            kvQuantizationBits: kvQuantizationBits
        )
    }

    // MARK: - Batch (AsyncStream)

    /// Estimate memory for multiple models concurrently, streaming results as they complete.
    ///
    /// Results arrive in completion order (fastest model first). Each result carries
    /// its `modelID`, so the caller can identify which model it belongs to.
    ///
    /// The stream respects task cancellation: cancelling the consuming task
    /// cancels all in-flight fetches.
    ///
    /// - Parameters:
    ///   - modelIDs: Array of HuggingFace model IDs.
    ///   - contextLength: Number of tokens for KV cache estimation. When `nil`,
    ///     each model uses its own `max_position_embeddings`, falling back to 4096.
    ///   - kvQuantizationBits: Optional KV cache quantization bits (e.g. 4 or 8).
    /// - Returns: An `AsyncStream` of results. Individual failures are `.failure` cases,
    ///   not thrown errors — the stream always emits one element per model ID.
    public func estimateMemory(
        for modelIDs: [String],
        contextLength: Int? = nil,
        kvQuantizationBits: Int? = nil
    ) -> AsyncStream<Result<MemoryEstimate, Error>> {
        AsyncStream { continuation in
            let task = Task {
                await withTaskGroup(of: Result<MemoryEstimate, Error>.self) { group in
                    for modelID in modelIDs {
                        group.addTask {
                            do {
                                let estimate = try await self.estimateMemory(
                                    for: modelID,
                                    contextLength: contextLength,
                                    kvQuantizationBits: kvQuantizationBits
                                )
                                return .success(estimate)
                            } catch {
                                return .failure(error)
                            }
                        }
                    }

                    for await result in group {
                        continuation.yield(result)
                    }
                }
                continuation.finish()
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}
