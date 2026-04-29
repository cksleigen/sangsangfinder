import Foundation

protocol NoticeRepository {
    func fetchNotices(query: String) async throws -> [Notice]
}
