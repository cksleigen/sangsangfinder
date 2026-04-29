import Foundation

@MainActor
final class NoticeViewModel: ObservableObject {
    @Published var notices: [Notice] = []
    @Published var reply: String = ""
    @Published var isLoading = false
    @Published var errorMessage: String? = nil

    private let useCase: NoticeSearchUseCase
    private var currentTask: Task<Void, Never>?

    init(useCase: NoticeSearchUseCase = NoticeSearchUseCaseImpl()) {
        self.useCase = useCase
    }

    func search(query: String, category: String? = nil) {
        let trimmed = query.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }

        // 진행 중인 요청 취소 후 새 요청 시작
        currentTask?.cancel()
        isLoading = true
        errorMessage = nil

        currentTask = Task {
            defer { isLoading = false }
            do {
                let result = try await useCase.search(query: trimmed, category: category)
                guard !Task.isCancelled else { return }
                notices = result.notices
                reply   = result.reply
            } catch is CancellationError {
                // 새 검색으로 취소된 요청 — 상태 변경 없음
            } catch {
                guard !Task.isCancelled else { return }
                errorMessage = error.localizedDescription
            }
        }
    }

    func clear() {
        currentTask?.cancel()
        notices      = []
        reply        = ""
        errorMessage = nil
    }
}
