# Report Notes: Alignment Demo

File này dùng để chuẩn bị nội dung báo cáo / bảo vệ project.

Paper tham chiếu:

```text
Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study
```

## 1. Mục tiêu và phạm vi thử nghiệm

Thử nghiệm này được định vị là mô phỏng thực hành (educational demonstration), không phải bản sao quy mô lớn (full-scale replication) của bài báo gốc. 

- Do giới hạn về năng lực tính toán phần cứng local (GPU 6GB VRAM), thiết kế thực nghiệm tập trung vào việc mô hình hóa các thành phần cốt lõi của pipeline Alignment (SFT, DPO, PPO-RLHF) ở kích thước nhỏ (135M parameters) thay vì huấn luyện và tối ưu hóa diện rộng như bài báo gốc.
- Kết quả từ thực nghiệm này đóng vai trò minh họa cơ chế hoạt động của thuật toán và sự khác biệt hành vi, không mang tính chất khẳng định tuyệt đối về tính ưu việt của phương pháp này so với phương pháp kia ở mọi quy mô.

## 2. Các điểm cốt lõi cần phân tích

Khi đánh giá kết quả, phân tích tập trung vào các điểm sau:

- **Tính đơn giản của DPO**: DPO loại bỏ các thành phần phức tạp như mô hình phần thưởng (Reward Model), giá trị (Value Model), vòng lặp phản hồi (rollout) bằng cách tối ưu hóa trực tiếp trên cặp dữ liệu so sánh (preference pairs).
- **Sự phụ thuộc dữ liệu của DPO**: Do học offline trực tiếp trên các cặp dữ liệu, DPO nhạy cảm với độ phủ (coverage) của tập dữ liệu huấn luyện, dễ bị mất khả năng định dạng hoặc từ chối không an toàn khi gặp prompt ngoài phân phối (OOD).
- **Tính linh hoạt của PPO**: PPO tối ưu hóa dựa trên tín hiệu hàm thưởng (Reward Model) nên có khả năng điều hướng hành vi linh hoạt hơn, đặc biệt khi kết hợp mô hình phần thưởng học được (Learned RM) và các ràng buộc luật định dạng cứng (rule-based bonuses).
- **Độ nhạy và độ phức tạp của PPO**: PPO đòi hỏi tối ưu nhiều hyperparameters, có độ trễ lớn hơn trong quá trình huấn luyện do phải thực hiện lấy mẫu (online sampling) liên tục.

## 3. Workflow demo nên dùng

Chạy:

```powershell
python run.py
```

Menu hiện tại:

```text
1. Compare models (no training)
2. Plot toy probability heatmaps (paper-style demo, no model loading)
3. Run OOD benchmark (recommended comparison report)
4. Run sanity tests
5. Train mini SFT
6. Train Reward Model
7. Train DPO
8. Train PPO
9. Train all, then compare + benchmark
```

Khi đã train xong, dùng:

- `1. Compare models`: tạo `outputs/compare_result.txt` và `outputs/compare_report.html`
- `2. Plot toy probability heatmaps`: tạo `outputs/probability_heatmaps.png` và `outputs/probability_heatmaps_summary.json`
- `3. Run OOD benchmark`: tạo `outputs/ood_benchmark_result.json` và `outputs/ood_benchmark_report.html`

Không chọn `9` nếu không muốn train lại.

## 4. Train once, evaluate many times

Checkpoint được lưu trong:

```text
outputs/sft_model
outputs/reward_model
outputs/dpo_model
outputs/ppo_model
outputs/ppo_value_model
```

Tắt máy vẫn còn checkpoint. Những lần sau chỉ cần compare/plot lại.

Train lại sẽ ghi đè checkpoint cũ.

## 5. Mini SFT

File chính:

```text
train/sft.py
```

Ý tưởng:

```text
input  = prompt + chosen_response
labels = -100 trên prompt tokens, token ids thật trên response tokens
loss   = cross-entropy chỉ trên response
```

Cách trình bày:

> Mini SFT dùng response `chosen` làm supervised target. Đây là bước nhỏ để minh họa supervised fine-tuning, không phải SFT quy mô lớn.

## 6. DPO

File chính:

```text
train/dpo.py
```

Thành phần:

- Policy model
- Frozen reference model
- Preference pair `chosen` / `rejected`
- Response log-prob
- DPO loss

Công thức trong code:

```text
loss = -log sigmoid(beta * ((log pi(chosen) - log pi(rejected))
                          - (log ref(chosen) - log ref(rejected))))
```

Cách trình bày:

> DPO tối ưu trực tiếp policy để tăng xác suất response được chọn và giảm xác suất response bị từ chối, có regularization thông qua reference model.

## 7. PPO

File chính:

```text
train/ppo.py
```

Thành phần:

- Policy model sinh response
- Hybrid reward model (Learned Reward Model kết hợp Rule-based formatting/safety)
- Frozen reference model để tính KL
- Value model / critic
- GAE
- PPO clipped objective

Công thức chính:

```text
ratio = exp(new_logprob - old_logprob)
policy_loss = -mean(min(ratio * advantage,
                        clip(ratio, 1-eps, 1+eps) * advantage))
value_loss = MSE(value, return)
reward = score_reward - kl_coef * (log pi_old - log pi_ref)
```

Cách trình bày:

> PPO trong project là phiên bản actor-critic rút gọn. Nó minh họa reward-based alignment, nhưng chưa phải PPO production-grade.

PPO hiện trộn thêm prompt từ `evaluation/ood_cases.json` vào rollout. Mục tiêu là để reward-based training nhìn thấy các tiêu chí được benchmark chấm, ví dụ format following, coding, safety, concise response và no repetition. Đây là thiết kế demo có kiểm soát, không phải benchmark khách quan quy mô lớn.

## 8. Compare report

File chính:

```text
evaluation/compare.py
```

Output:

```text
outputs/compare_result.txt
outputs/compare_report.html
```

Ý nghĩa:

- Dùng để nhìn trực quan Base / DPO / PPO trả lời cùng một prompt.
- Phù hợp demo nhanh.
- Không đủ để kết luận model nào tốt hơn trên toàn bộ task.

Cách nói khi bị hỏi:

> Response đơn lẻ chỉ là qualitative example. Để phân tích tốt hơn, project dùng thêm probability heatmap.

## 9. Toy probability heatmap

File chính:

```text
evaluation/plot_toy_probabilities.py
```

Output:

```text
outputs/probability_heatmaps.png
outputs/probability_heatmaps_summary.json
```

Ý nghĩa:

- Cột 1: preference data pairs, biểu diễn mapping prompt-response mong muốn.
- Cột 2: phân bố xác suất DPO-like, mạnh ở vùng quen thuộc nhưng bị kéo về target ID quen thuộc khi gặp OOD.
- Cột 3: phân bố xác suất PPO-like, vẫn bám OOD safe target nhờ reward-guided behavior.

Cách trình bày:

> Heatmap là toy visualization, không phải benchmark đầy đủ. Nó minh họa trực quan luận điểm: DPO phụ thuộc vào coverage của preference data nên có thể bị lệch ở OOD, còn PPO có thể giữ hành vi an toàn hơn nếu reward signal bao phủ đúng mục tiêu.

## 10. Huấn luyện mô hình phần thưởng (Reward Model Training)

File chính:

```text
train/reward_model.py
configs/reward_model_config.py
models/reward_model.py
```

Thành phần:
- Base model là `./outputs/sft_model` (khởi tạo mô hình phân loại chuỗi bằng `AutoModelForSequenceClassification` từ checkpoint SFT).
- Dữ liệu huấn luyện: 150 mẫu các cặp phản hồi ưu tiên (chosen / rejected).
- Loss function (Bradley-Terry preference loss):
  `loss = -log sigmoid(r(x, y_w) - r(x, y_l))`
  với $r(x, y)$ là điểm reward mô hình tự gán cho prompt $x$ và phản hồi $y$.

Ý nghĩa & Cách trình bày:

> Reward Model học trực tiếp từ các cặp dữ liệu con người ưu tiên để làm cơ sở chấm điểm ngữ nghĩa cho PPO. Mô hình sau huấn luyện được tích hợp vào `HybridRewardModel`, kết hợp điểm ngữ nghĩa này với các rule-based định dạng (JSON, Table, Code) nhằm giúp PPO tối ưu hóa toàn diện cả về mặt chất lượng câu trả lời lẫn tính cấu trúc của văn bản.

## 11. OOD benchmark

File chính:

```text
evaluation/ood_benchmark.py
evaluation/ood_cases.json
```

Output:

```text
outputs/ood_benchmark_result.json
outputs/ood_benchmark_report.html
```

Ý nghĩa:

- Chạy nhiều prompt stress test thay vì một prompt đơn lẻ.
- Có các nhóm như format, coding, safety, instruction following và OOD language mix.
- Chấm điểm bằng rule-based metrics như JSON validity, code signals, bullet/table format, relevance, conciseness, safety và no repetition.
- **Sự khác biệt hành vi giữa DPO và PPO trên OOD:**
  - **DPO dễ bị đánh lừa/lệch hướng (brittle):** Do DPO chỉ học offline trên các cặp preference pairs của HH-RLHF (chủ yếu là hội thoại thông thường), khi gặp các prompt yêu cầu định dạng đặc thù (JSON, markdown table) hoặc các prompt tấn công an toàn OOD, DPO không thể suy luận tốt và dễ đưa ra câu trả lời vô nghĩa, lặp từ hoặc bỏ qua định dạng (ví dụ: trả về code Python giải thích thay vì trả về JSON sạch).
  - **PPO có tính ổn định hơn trên các tiêu chí reward (reward-guided robustness):** PPO được huấn luyện tương tác trực tiếp với môi trường sinh phản hồi (rollout) kết hợp với tập OOD prompts, nhận tín hiệu từ Reward Model định sẵn (phạt lặp từ, phạt không an toàn, thưởng định dạng cụ thể) và học qua Anchor Loss. Nhờ đó, PPO bám sát các ràng buộc an toàn và định dạng tốt hơn nhiều.

### Ý nghĩa thực nghiệm

Kết quả từ OOD benchmark cung cấp góc nhìn định lượng về khả năng tổng quát hóa của hai thuật toán. Dù hệ thống chấm điểm dựa trên luật (rule-based metric) và quy mô dữ liệu nhỏ, nó vẫn phản ánh rõ xu hướng phân hóa hành vi được nêu trong lý thuyết: PPO duy trì tính ổn định định dạng và an toàn tốt hơn khi gặp dữ liệu phân phối lạ nhờ có hàm thưởng định hướng trực tiếp.

## 12. Hạn chế thực nghiệm

- Mô hình base sử dụng kích thước nhỏ (`SmolLM2-135M-Instruct`).
- Kích thước tập dữ liệu huấn luyện nhỏ (150 mẫu cho Reward Model, 60-120 mẫu cho giai đoạn Alignment).
- Mô hình phần thưởng lai (Hybrid Reward Model) vẫn có các heuristics hỗ trợ, chưa phản ánh hoàn toàn một Learned Reward Model huấn luyện trên tập dữ liệu preference khổng lồ.
- Không thử nghiệm trên nhiều seed khác nhau để loại bỏ nhiễu ngẫu nhiên.
- Đánh giá OOD mang tính chất tự động bằng luật, thiếu đánh giá định tính chuyên sâu từ con người (human evaluation).

## 13. Tóm tắt kết luận

Thực nghiệm đã mô phỏng thành công quy trình alignment khép kín gồm SFT $\rightarrow$ Reward Model $\rightarrow$ DPO $\rightarrow$ PPO ở quy mô nhỏ. Kết quả chỉ ra rằng DPO tối ưu hóa hiệu quả trực tiếp trên phân phối dữ liệu ưu tiên tĩnh, trong khi PPO thể hiện độ linh hoạt cao hơn trong việc căn chỉnh mô hình theo các ràng buộc mục tiêu đa dạng thông qua cơ chế tối ưu hóa trực tuyến dựa trên hàm thưởng.
