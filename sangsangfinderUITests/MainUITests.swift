import XCTest

final class MainUITests: XCTestCase {
    let app = XCUIApplication()

    override func setUpWithError() throws {
        continueAfterFailure = false
        app.launch()
    }

    func test_appLaunches() {
        XCTAssertTrue(app.navigationBars["상상파인더"].exists)
    }
}
