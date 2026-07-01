# Report Notes: Alignment Demo

File này dùng để chuẩn bị nội dung báo cáo / bảo vệ project.

Paper tham chiếu:

```text
Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study
```

## 1. Cách định vị project

Project này là **educational demo**, không phải reproduction đầy đủ của paper.

Câu nên nói:

> Project triển khai các thành phần cốt lõi của mini SFT, DPO và PPO-RLHF ở quy mô nhỏ để minh họa cơ chế alignment. Do giới hạn laptop local, kết quả không dùng để kết luận tuyệt đối rằng DPO hay PPO luôn tốt hơn.

Câu không nên nói:

> Project chứng minh PPO tốt hơn DPO giống paper.

Lý do: paper dùng mô hình, dữ liệu, reward model và benchmark được tối ưu hơn nhiều. Demo local không đủ quy mô để kết luận như vậy.

## 2. Luận điểm chính khi trình bày

Nên tập trung vào:

- DPO học trực tiếp từ preference pairs `chosen` / `rejected`.
- DPO đơn giản hơn PPO vì không cần rollout, value model, reward loop phức tạp.
- DPO có thể bị giới hạn bởi coverage của preference data.
- PPO tối ưu qua reward signal, nên linh hoạt hơn nếu reward signal phù hợp.
- PPO cũng khó ổn định hơn và phụ thuộc mạnh vào reward design.

Kết luận hợp lý:

> Demo minh họa sự khác biệt cơ chế giữa DPO và PPO. DPO gần với preference optimization trực tiếp, còn PPO thể hiện reward-based alignment thông qua rollout, reward, KL penalty và value function.

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

Cách trình bày:

> Đây là benchmark demo có tiêu chí rõ ràng hơn so với xem một response đơn lẻ. Điểm số vẫn là rule-based, không phải human evaluation, nhưng phù hợp để minh họa sự khác biệt hành vi giữa Base, DPO và PPO trong phạm vi project local.

## 12. Hạn chế cần nói rõ

- Model nhỏ: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Dữ liệu train ít (chỉ dùng 150 mẫu cho Reward Model, 60-120 mẫu cho DPO/PPO)
- Reward Model còn nhỏ (135M tham số) và hàm reward lai vẫn có các heuristics hỗ trợ
- Không có benchmark nhiều seed
- Không đánh giá bằng human preference thật ở quy mô lớn
- Không đủ điều kiện tái hiện đầy đủ kết quả của paper

## 13. Kết luận nên dùng

Kết luận phù hợp:

> Project cho thấy pipeline alignment gồm mini SFT, DPO và PPO ở quy mô nhỏ. DPO đơn giản và gần với objective preference trực tiếp, còn PPO minh họa reward-based optimization với rollout, value model, KL penalty và clipped objective. Các kết quả compare/heatmap dùng để minh họa cơ chế, không phải kết luận benchmark tuyệt đối.
