import Foundation

protocol NoticeSearchUseCase {
    func search(query: String) async throws -> [Notice]
}
