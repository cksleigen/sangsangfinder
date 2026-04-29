import SwiftUI

// MARK: - Card

struct NoticeCardView: View {
    let notice: Notice
    let index: Int

    @State private var appeared = false
    @Environment(\.openURL) private var openURL

    var body: some View {
        Button {
            if let url = URL(string: notice.url) { openURL(url) }
        } label: {
            cardContent
        }
        .buttonStyle(.plain)
        .offset(y: appeared ? 0 : 24)
        .opacity(appeared ? 1 : 0)
        .onAppear {
            withAnimation(
                .spring(response: 0.5, dampingFraction: 0.78)
                .delay(Double(index) * 0.055)
            ) { appeared = true }
        }
    }

    private var cardContent: some View {
        HStack(spacing: 0) {
            // 카테고리 색상 왼쪽 스트라이프
            RoundedRectangle(cornerRadius: 3)
                .fill(notice.category.categoryColor)
                .frame(width: 4)
                .padding(.vertical, 14)

            VStack(alignment: .leading, spacing: 8) {
                // 카테고리 뱃지 + 날짜
                HStack {
                    CategoryBadge(category: notice.category)
                    Spacer()
                    Text(notice.date)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }

                // 제목
                Text(notice.title)
                    .font(.system(.subheadline, weight: .semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)

                // 요약 (있을 때)
                if let summary = notice.summary, !summary.isEmpty {
                    Text(summary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }

                // 바로가기 링크
                HStack {
                    Spacer()
                    HStack(spacing: 3) {
                        Text("공지 바로가기")
                            .font(.caption)
                            .fontWeight(.medium)
                        Image(systemName: "arrow.up.right")
                            .font(.caption2)
                    }
                    .foregroundStyle(notice.category.categoryColor)
                }
            }
            .padding(.leading, 14)
            .padding(.trailing, 16)
            .padding(.vertical, 14)
        }
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .shadow(color: .black.opacity(0.06), radius: 10, x: 0, y: 3)
    }
}

// MARK: - Category Badge

struct CategoryBadge: View {
    let category: String

    var body: some View {
        Text(category)
            .font(.system(size: 11, weight: .bold))
            .foregroundStyle(category.categoryColor)
            .padding(.horizontal, 9)
            .padding(.vertical, 3)
            .background(category.categoryColor.opacity(0.12))
            .clipShape(Capsule())
    }
}

// MARK: - Category Color

extension String {
    var categoryColor: Color {
        switch self {
        case "취업/채용":       return Color(red: 0.20, green: 0.78, blue: 0.35)  // 초록
        case "인턴십":          return Color(red: 1.00, green: 0.58, blue: 0.00)  // 주황
        case "장학금":          return Color(red: 0.90, green: 0.68, blue: 0.00)  // 황금
        case "학자금/근로장학":  return Color(red: 0.00, green: 0.78, blue: 0.64)  // 민트
        case "학사행정":        return Color(red: 0.35, green: 0.34, blue: 0.84)  // 인디고
        case "창업":            return Color(red: 1.00, green: 0.27, blue: 0.27)  // 빨강
        case "국제교류":        return Color(red: 0.20, green: 0.67, blue: 0.86)  // 하늘
        case "교육/특강":       return Color(red: 0.04, green: 0.52, blue: 1.00)  // 파랑
        case "비교과":          return Color(red: 0.69, green: 0.32, blue: 0.87)  // 보라
        case "공모전/경진대회":  return Color(red: 1.00, green: 0.18, blue: 0.55)  // 핫핑크
        case "봉사/서포터즈":   return Color(red: 0.19, green: 0.82, blue: 0.58)  // 에메랄드
        case "기숙사/생활관":   return Color(red: 0.64, green: 0.44, blue: 0.26)  // 브라운
        case "ROTC":            return Color(red: 0.50, green: 0.50, blue: 0.50)  // 회색
        default:                return Color(red: 0.56, green: 0.56, blue: 0.58)
        }
    }
}
