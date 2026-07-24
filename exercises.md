# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> *Khi temperature thấp (0.0), model thường trả lời ổn định, ít thay đổi giữa các lần chạy và tập trung vào thông tin chính xác. Khi tăng temperature lên 1.0 hoặc 1.5, câu trả lời có xu hướng đa dạng hơn, cách diễn đạt sáng tạo hơn nhưng đôi khi có thể thêm các chi tiết không cần thiết. Temperature càng cao thì mức độ ngẫu nhiên trong quá trình sinh câu trả lời càng lớn.*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *Với chatbot hỗ trợ khách hàng, tôi sẽ chọn temperature khoảng 0.2–0.4. Lý do là chatbot cần trả lời nhất quán, chính xác theo thông tin sản phẩm và hạn chế tạo ra thông tin sai lệch. Temperature cao phù hợp hơn với các tác vụ sáng tạo như viết nội dung hoặc brainstorming.*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> *Với cùng một lượng token đầu ra, GPT-4o có chi phí cao hơn GPT-4o-mini khoảng nhiều lần vì giá token của GPT-4o cao hơn. Với workload lớn như 10.000 người dùng mỗi ngày và mỗi người gọi nhiều lần, sử dụng GPT-4o-mini sẽ tiết kiệm đáng kể chi phí.

GPT-4o phù hợp với các trường hợp yêu cầu chất lượng suy luận cao như phân tích chuyên sâu, viết code phức tạp hoặc xử lý nghiệp vụ quan trọng. GPT-4o-mini phù hợp với chatbot thông thường, phân loại dữ liệu hoặc các tác vụ có số lượng request lớn.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Khi sử dụng system prompt là giáo viên tiểu học, model sử dụng ngôn ngữ đơn giản hơn, giải thích bằng ví dụ gần gũi và tránh nhiều thuật ngữ kỹ thuật. Khi sử dụng persona chuyên gia tài chính, câu trả lời dài hơn, sử dụng nhiều thuật ngữ chuyên ngành và phân tích sâu hơn.

System prompt có ảnh hưởng lớn đến cách model lựa chọn từ ngữ, mức độ chi tiết và phong cách phản hồi. Nó giúp định hướng hành vi của model thay vì thay đổi kiến thức bên trong model.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> *Khi so sánh một đoạn văn tiếng Việt khoảng 100 từ, số token do tiktoken trả về thường cao hơn so với cách ước lượng số từ / 0.75. Nguyên nhân là tokenizer không chỉ dựa vào số từ mà còn dựa vào cách chia nhỏ ký tự, từ và dấu câu.

Tiếng Việt thường tốn nhiều token hơn tiếng Anh vì nhiều từ tiếng Việt có dấu và cấu trúc tách token khác với tiếng Anh. Một câu có cùng số lượng từ nhưng có thể tạo ra số token khác nhau.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming hữu ích nhất khi model tạo ra câu trả lời dài như chatbot, trợ lý AI, viết nội dung hoặc phân tích tài liệu. Người dùng có thể nhìn thấy phản hồi xuất hiện từng phần thay vì phải chờ toàn bộ kết quả, giúp trải nghiệm nhanh và tự nhiên hơn.

Non-streaming phù hợp với các tác vụ cần lấy toàn bộ kết quả trước khi xử lý tiếp như gọi API nội bộ, tạo báo cáo tự động hoặc khi response rất ngắn.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *Exponential backoff giúp giảm áp lực lên hệ thống khi API đang quá tải bằng cách tăng dần thời gian chờ giữa các lần thử lại. So với delay cố định, cách này giúp các client không gửi lại request cùng lúc.

Nếu hàng nghìn client cùng retry sau đúng 1 giây, hệ thống có thể xảy ra hiện tượng "thundering herd", tức là rất nhiều request quay lại cùng thời điểm làm server tiếp tục quá tải.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *Persona tôi chọn là một trợ lý học tập AI.
System prompt:
"Bạn là một trợ lý học tập AI.
Hãy giải thích các khái niệm công nghệ bằng tiếng Việt,
ưu tiên cách giải thích dễ hiểu cho người mới bắt đầu.
Trả lời có cấu trúc, ngắn gọn và đưa ví dụ thực tế khi cần."
Tôi sử dụng cụm "người mới bắt đầu" để yêu cầu model tránh sử dụng quá nhiều thuật ngữ khó. Cụm "trả lời có cấu trúc" giúp câu trả lời dễ đọc hơn và phù hợp với việc học tập.
*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> *Hạn chế lớn nhất hiện tại là history chỉ lưu được một số lượt hội thoại gần nhất nên trợ lý có thể quên các thông tin ở những cuộc hội thoại dài.

Một cải thiện có thể triển khai là thêm bộ nhớ dài hạn bằng cách lưu thông tin quan trọng vào database hoặc vector database. Khi người dùng hỏi lại, hệ thống có thể tìm kiếm thông tin liên quan và đưa vào context trước khi gọi model.*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
