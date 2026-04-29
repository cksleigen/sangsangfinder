import Foundation

// MARK: - Response (FastAPI schemas.py: HealthResponse)

struct HealthResponse: Decodable {
    let status: String
    let noticesCount: Int
    let indexedCount: Int

    enum CodingKeys: String, CodingKey {
        case status
        case noticesCount = "notices_count"
        case indexedCount = "indexed_count"
    }

    var isReady: Bool { status == "ok" && indexedCount > 0 }
}
