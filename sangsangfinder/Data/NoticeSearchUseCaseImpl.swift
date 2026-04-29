import Foundation

struct NoticeSearchUseCaseImpl: NoticeSearchUseCase {
    private let apiService: NoticeAPIService

    init(apiService: NoticeAPIService = NoticeAPIService()) {
        self.apiService = apiService
    }

    func search(query: String, category: String?) async throws -> SearchResult {
        let request = SearchRequest(query: query, category: category)
        let response = try await apiService.search(request: request)
        return SearchResult(reply: response.reply, notices: response.notices)
    }
}
