import Foundation

/// Errors thrown by MLXLLMChecker operations.
public enum MLXLLMCheckerError: Error, LocalizedError, Sendable {
    case invalidModelID(String)
    case configNotFound(String)
    case configParsingFailed(String, underlying: Error)
    case missingRequiredField(String, model: String)
    case networkError(String, underlying: Error)

    public var errorDescription: String? {
        switch self {
        case .invalidModelID(let id):
            return "Invalid model ID: '\(id)'. Expected format: 'namespace/model-name'."
        case .configNotFound(let id):
            return "config.json not found for model '\(id)'."
        case .configParsingFailed(let id, let underlying):
            return "Failed to parse config.json for '\(id)': \(underlying.localizedDescription)"
        case .missingRequiredField(let field, let model):
            return "Required field '\(field)' missing in config.json for '\(model)'."
        case .networkError(let id, let underlying):
            return "Network error fetching model '\(id)': \(underlying.localizedDescription)"
        }
    }
}
