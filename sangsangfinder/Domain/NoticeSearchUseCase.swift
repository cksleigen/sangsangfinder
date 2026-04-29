import Foundation

protocol NoticeSearchUseCase {
    func search(query: String, category: String?) async throws -> SearchResult
}
