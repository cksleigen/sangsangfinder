import SwiftUI

struct SearchView: View {
    @StateObject private var viewModel = NoticeViewModel()
    @State private var searchText = ""

    var body: some View {
        NavigationStack {
            ZStack {
                Color(.systemGroupedBackground).ignoresSafeArea()
                contentLayer
            }
            .navigationTitle("상상파인더")
            .navigationBarTitleDisplayMode(.large)
            .toolbar { clearToolbarItem }
        }
        .searchable(
            text: $searchText,
            placement: .navigationBarDrawer(displayMode: .always),
            prompt: "장학금, 수강신청, 취업박람회..."
        )
        .onSubmit(of: .search) {
            viewModel.search(query: searchText)
        }
    }

    // MARK: - State Router

    @ViewBuilder
    private var contentLayer: some View {
        switch viewState {
        case .prompt:   promptView
        case .loading:  loadingView
        case .error:    errorView
        case .results:  resultsList
        }
    }

    private enum ViewState { case prompt, loading, error, results }

    private var viewState: ViewState {
        if viewModel.isLoading                                    { return .loading }
        if viewModel.errorMessage != nil                          { return .error }
        if viewModel.notices.isEmpty && viewModel.reply.isEmpty   { return .prompt }
        return .results
    }

    // MARK: - Results

    private var resultsList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 10) {
                // 결과 카운트
                HStack {
                    Text("검색 결과 \(viewModel.notices.count)건")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
                .padding(.horizontal, 4)
                .padding(.bottom, 2)

                // AI 답변 카드
                if !viewModel.reply.isEmpty {
                    replyCard
                }

                // 공지 카드 목록
                ForEach(Array(viewModel.notices.enumerated()), id: \.element.id) { index, notice in
                    NoticeCardView(notice: notice, index: index)
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 12)
            .padding(.bottom, 32)
        }
    }

    // MARK: - AI 답변 카드

    private var replyCard: some View {
        HStack(alignment: .top, spacing: 12) {
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [Color.blue, Color.purple.opacity(0.8)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 34, height: 34)
                Image(systemName: "sparkles")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.white)
            }

            VStack(alignment: .leading, spacing: 5) {
                Text("AI 답변")
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundStyle(
                        LinearGradient(
                            colors: [.blue, .purple],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                Text(viewModel.reply)
                    .font(.subheadline)
                    .foregroundStyle(.primary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(
                            LinearGradient(
                                colors: [Color.blue.opacity(0.4), Color.purple.opacity(0.3)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 1
                        )
                )
        )
        .shadow(color: .blue.opacity(0.08), radius: 10, x: 0, y: 3)
        .transition(.move(edge: .top).combined(with: .opacity))
    }

    // MARK: - Empty Prompt

    private var promptView: some View {
        VStack(spacing: 28) {
            Spacer()

            VStack(spacing: 14) {
                ZStack {
                    Circle()
                        .fill(Color.blue.opacity(0.08))
                        .frame(width: 88, height: 88)
                    Image(systemName: "magnifyingglass")
                        .font(.system(size: 36, weight: .light))
                        .foregroundStyle(Color.blue.opacity(0.5))
                }

                VStack(spacing: 6) {
                    Text("공지사항을 검색해보세요")
                        .font(.headline)
                        .foregroundStyle(.secondary)
                    Text("자연어로 물어보면 AI가 답변해드려요")
                        .font(.subheadline)
                        .foregroundStyle(.tertiary)
                }
            }

            // 추천 검색어 칩
            VStack(spacing: 10) {
                Text("이런 것도 검색해보세요")
                    .font(.caption)
                    .foregroundStyle(.quaternary)

                FlowLayout(spacing: 8) {
                    ForEach(["장학금 신청", "수강신청 기간", "취업박람회", "기숙사 입사", "교환학생"], id: \.self) { suggestion in
                        Button(suggestion) {
                            searchText = suggestion
                            viewModel.search(query: suggestion)
                        }
                        .font(.caption)
                        .fontWeight(.medium)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 7)
                        .background(Color(.secondarySystemGroupedBackground))
                        .foregroundStyle(.secondary)
                        .clipShape(Capsule())
                    }
                }
            }

            Spacer()
        }
        .padding(.horizontal, 32)
    }

    // MARK: - Loading

    private var loadingView: some View {
        VStack(spacing: 14) {
            ProgressView()
                .scaleEffect(1.3)
                .tint(.blue)
            Text("검색 중...")
                .font(.subheadline)
                .foregroundStyle(.tertiary)
        }
    }

    // MARK: - Error

    private var errorView: some View {
        VStack(spacing: 20) {
            Image(systemName: "wifi.exclamationmark")
                .font(.system(size: 44, weight: .light))
                .foregroundStyle(.orange.opacity(0.7))

            VStack(spacing: 6) {
                Text("연결에 실패했습니다")
                    .font(.headline)
                    .foregroundStyle(.secondary)
                if let msg = viewModel.errorMessage {
                    Text(msg)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.center)
                }
            }

            Button {
                viewModel.search(query: searchText)
            } label: {
                Label("다시 시도", systemImage: "arrow.clockwise")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(Color.blue.opacity(0.1))
                    .foregroundStyle(.blue)
                    .clipShape(Capsule())
            }
        }
        .padding(32)
    }

    // MARK: - Toolbar

    @ToolbarContentBuilder
    private var clearToolbarItem: some ToolbarContent {
        ToolbarItem(placement: .topBarTrailing) {
            if !viewModel.notices.isEmpty {
                Button {
                    withAnimation(.easeOut(duration: 0.2)) {
                        viewModel.clear()
                        searchText = ""
                    }
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.tertiary)
                }
            }
        }
    }
}

// MARK: - FlowLayout (추천 검색어 줄바꿈)

struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? 0
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0

        for view in subviews {
            let size = view.sizeThatFits(.unspecified)
            if x + size.width > width, x > 0 {
                y += rowHeight + spacing
                x = 0
                rowHeight = 0
            }
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
        return CGSize(width: width, height: y + rowHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX
        var y = bounds.minY
        var rowHeight: CGFloat = 0

        for view in subviews {
            let size = view.sizeThatFits(.unspecified)
            if x + size.width > bounds.maxX, x > bounds.minX {
                y += rowHeight + spacing
                x = bounds.minX
                rowHeight = 0
            }
            view.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}
