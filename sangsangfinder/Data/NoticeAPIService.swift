import Foundation

enum APIError: LocalizedError {
    case invalidURL
    case httpError(Int)
    case decodingFailed(Error)

    var errorDescription: String? {
        switch self {
        case .invalidURL:            return "잘못된 서버 주소입니다."
        case .httpError(let code):   return "서버 오류 (HTTP \(code))"
        case .decodingFailed:        return "응답 파싱에 실패했습니다."
        }
    }
}

struct NoticeAPIService {
    private let baseURL: String
    private let session: URLSession

    init(baseURL: String = Constants.API.baseURL, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.session = session
    }

    func search(request: SearchRequest) async throws -> SearchResponse {
        try await post(path: "/api/v1/search", body: request)
    }

    func recommend(request: RecommendRequest) async throws -> RecommendResponse {
        try await post(path: "/api/v1/recommend", body: request)
    }

    func health() async throws -> HealthResponse {
        guard let url = URL(string: "\(baseURL)/api/v1/health") else {
            throw APIError.invalidURL
        }
        let (data, response) = try await session.data(from: url)
        try validate(response)
        return try decode(data)
    }
}

// MARK: - Private

private extension NoticeAPIService {
    func post<Body: Encodable, Response: Decodable>(
        path: String,
        body: Body
    ) async throws -> Response {
        guard let url = URL(string: "\(baseURL)\(path)") else {
            throw APIError.invalidURL
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await session.data(for: request)
        try validate(response)
        return try decode(data)
    }

    func validate(_ response: URLResponse) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            throw APIError.httpError(http.statusCode)
        }
    }

    func decode<T: Decodable>(_ data: Data) throws -> T {
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw APIError.decodingFailed(error)
        }
    }
}
