import Foundation

// MARK: - DTO (FastAPI schemas.py: NoticeItem)

struct NoticeItemDTO: Decodable {
    let title: String
    let url: String
    let date: String
    let category: String
    let score: Double
    let summary: String?

    // 필드명이 모두 단일 소문자 단어라 CodingKeys 불필요
}

// MARK: - Domain Mapping

extension NoticeItemDTO {
    func toDomain() -> Notice {
        Notice(
            id:       url,   // API에 별도 id 없음 — URL이 고유 식별자
            title:    title,
            category: category,
            date:     date,
            url:      url,
            score:    score,
            summary:  summary
        )
    }
}
