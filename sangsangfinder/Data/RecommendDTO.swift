import Foundation

// MARK: - Request (FastAPI schemas.py: RecommendRequest)

struct RecommendRequest: Encodable {
    let college: String
    let track: String
    let grade: String
    let interests: [String]
    let topK: Int

    enum CodingKeys: String, CodingKey {
        case college, track, grade, interests
        case topK = "top_k"
    }

    init(
        college: String,
        track: String,
        grade: String,
        interests: [String] = [],
        topK: Int = 5
    ) {
        self.college   = college
        self.track     = track
        self.grade     = grade
        self.interests = interests
        self.topK      = topK
    }
}

// MARK: - Response (FastAPI schemas.py: RecommendResponse)

struct RecommendResponse: Decodable {
    let results: [NoticeItemDTO]
}

extension RecommendResponse {
    var notices: [Notice] { results.map { $0.toDomain() } }
}
