import XCTest
@testable import sangsangfinder

final class NoticeEntityTests: XCTestCase {
    func test_notice_init() {
        let notice = Notice(
            id: "1",
            title: "테스트 공지",
            category: "학사",
            date: Date(),
            url: "https://example.com"
        )
        XCTAssertEqual(notice.id, "1")
        XCTAssertEqual(notice.title, "테스트 공지")
    }
}
