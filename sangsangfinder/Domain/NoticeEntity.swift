import Foundation

struct Notice {
    let id: String
    let title: String
    let category: String
    let date: String    // API가 "2026.04.29" 형식 문자열로 반환
    let url: String
    let score: Double
    let summary: String?
}

struct SearchResult {
    let reply: String
    let notices: [Notice]
}
