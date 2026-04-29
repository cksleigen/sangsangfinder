import Foundation

// MARK: - Request (FastAPI schemas.py: SearchRequest)

struct SearchRequest: Encodable {
    let query: String
    let category: String?
    let topK: Int
    let alpha: Double
    let isFirst: Bool

    enum CodingKeys: String, CodingKey {
        case query, category, alpha
        case topK    = "top_k"
        case isFirst = "is_first"
    }

    init(
        query: String,
        category: String? = nil,
        topK: Int = 5,
        alpha: Double = 0.7,
        isFirst: Bool = false
    ) {
        self.query    = query
        self.category = category
        self.topK     = topK
        self.alpha    = alpha
        self.isFirst  = isFirst
    }
}

// MARK: - Response (FastAPI schemas.py: SearchResponse)

struct SearchResponse: Decodable {
    let reply: String
    let results: [NoticeItemDTO]
}

extension SearchResponse {
    var notices: [Notice] { results.map { $0.toDomain() } }
}
