import Foundation
import HuggingFace

/// Fetches model configuration and file metadata from HuggingFace Hub.
struct HubModelFetcher: Sendable {

    let client: HubClient

    /// Data fetched for a single model.
    struct FetchResult: Sendable {
        let config: ModelConfig
        let safetensorsBytes: UInt64?
        let totalDownloadBytes: UInt64?
    }

    /// Fetch config.json and safetensors file sizes concurrently.
    ///
    /// - Parameter modelID: HuggingFace model ID (e.g. "mlx-community/Qwen3-4B-4bit").
    /// - Returns: Parsed config and total safetensors weight size.
    func fetch(modelID: String) async throws -> FetchResult {
        guard let repoID = Repo.ID(rawValue: modelID) else {
            throw MLXLLMCheckerError.invalidModelID(modelID)
        }

        // Fetch config.json and file listing concurrently
        async let configData = client.downloadContentsOfFile(
            at: "config.json",
            from: repoID
        )
        async let fileEntries = client.listFiles(
            in: repoID,
            kind: .model,
            recursive: true
        )

        let data: Data
        do {
            data = try await configData
        } catch {
            throw MLXLLMCheckerError.networkError(modelID, underlying: error)
        }

        let config: ModelConfig
        do {
            config = try JSONDecoder().decode(ModelConfig.self, from: data)
        } catch {
            throw MLXLLMCheckerError.configParsingFailed(modelID, underlying: error)
        }

        // Sum file sizes (best-effort, don't fail if listing fails)
        let safetensorsBytes: UInt64?
        let totalDownloadBytes: UInt64?
        do {
            let entries = try await fileEntries
            let files = entries.filter { $0.type == .file }

            let safetensorsSize = files
                .filter { $0.path.hasSuffix(".safetensors") }
                .compactMap(\.size)
                .reduce(0, +)
            safetensorsBytes = safetensorsSize > 0 ? UInt64(safetensorsSize) : nil

            let totalSize = files
                .compactMap(\.size)
                .reduce(0, +)
            totalDownloadBytes = totalSize > 0 ? UInt64(totalSize) : nil
        } catch {
            safetensorsBytes = nil
            totalDownloadBytes = nil
        }

        return FetchResult(
            config: config,
            safetensorsBytes: safetensorsBytes,
            totalDownloadBytes: totalDownloadBytes
        )
    }
}
