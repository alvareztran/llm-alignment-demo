# Alignment Demo: SFT, DPO, PPO

Project này là demo nhỏ cho môn Machine Learning / NLP về **LLM alignment**. Mục tiêu không phải tái hiện đầy đủ benchmark quy mô lớn, mà là giúp sinh viên nhìn rõ pipeline:

1. Mini SFT trên response `chosen`
2. DPO từ preference pairs `chosen` / `rejected`
3. PPO-RLHF rút gọn với reward, value model, KL penalty và clipped objective
4. So sánh output và vẽ toy probability heatmap để minh họa ý tưởng trong paper
5. Chạy OOD benchmark có scoring rule-based và HTML report

Paper tham chiếu:

```text
Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study
```

## 1. Thông điệp demo

Với laptop local và model nhỏ, không nên kết luận rằng DPO hoặc PPO luôn tốt hơn. Project này nên được trình bày như một **controlled educational demo**:

- DPO đơn giản hơn, học trực tiếp từ preference data.
- DPO phụ thuộc nhiều vào coverage của preference data.
- PPO phức tạp hơn, nhưng có thể tối ưu theo reward signal cụ thể.
- Khi gặp prompt / dữ liệu lạ, demo nên dùng stress test hoặc probability plot để minh họa sự khác nhau, thay vì chỉ nhìn một response đơn lẻ.

Câu nên dùng khi báo cáo:

> Do giới hạn tài nguyên local, project không tái hiện đầy đủ quy mô huấn luyện của paper. Thay vào đó, project minh họa các cơ chế chính của SFT, DPO và PPO, đồng thời cung cấp compare report và probability heatmap để hỗ trợ phân tích.

## 2. Cấu trúc project

```text
alignment-demo/
├── configs/
│   ├── dpo_config.py
│   ├── ppo_config.py
│   └── sft_config.py
├── data/
│   └── prepare_dataset.py
├── evaluation/
│   ├── compare.py
│   ├── generate.py
│   └── plot_probabilities.py
├── models/
│   ├── load_model.py
│   ├── policy_reference.py
│   ├── reward_model.py
│   └── value_model.py
├── ppo/
│   └── reward.py
├── train/
│   ├── dpo.py
│   ├── ppo.py
│   └── sft.py
├── utils/
│   ├── advantage.py
│   ├── logprob.py
│   └── sampling.py
├── outputs/
├── tests/
├── run.py
└── requirements.txt
```

## 3. File quan trọng

| File | Vai trò |
|---|---|
| `data/prepare_dataset.py` | Load `Anthropic/hh-rlhf`, tách `prompt`, `chosen`, `rejected` |
| `models/load_model.py` | Load tokenizer và model mặc định |
| `models/policy_reference.py` | Load policy model và frozen reference model |
| `train/sft.py` | Mini SFT bằng cross-entropy trên response `chosen` |
| `train/dpo.py` | DPO loss với policy/reference log-ratio |
| `train/ppo.py` | PPO actor-critic rút gọn |
| `ppo/reward.py` | Reward rule-based cho PPO demo |
| `models/value_model.py` | Value model / critic cho PPO |
| `utils/logprob.py` | Tính token log-prob cho causal LM |
| `utils/advantage.py` | Tính GAE và normalize advantage |
| `evaluation/compare.py` | Compare Base / DPO / PPO, xuất text và HTML |
| `evaluation/ood_cases.json` | Bộ prompt OOD/stress test cho benchmark |
| `evaluation/ood_benchmark.py` | Chạy benchmark rule-based và xuất HTML/JSON report |
| `evaluation/plot_toy_probabilities.py` | Vẽ toy heatmap kiểu paper-style, không load model |
| `train/reward_model.py` | Huấn luyện mô hình phần thưởng (Reward Model) từ các cặp preferences |
| `configs/reward_model_config.py` | Cấu hình cho huấn luyện mô hình phần thưởng |
| `evaluation/plot_probabilities.py` | Vẽ xác suất chéo prompt-to-response đã hiệu chỉnh (calibrated) và co giãn nhiệt độ |
| `run.py` | Menu chính để chạy compare, plot, test, train |

## 4. Model và dữ liệu

Model mặc định:

```text
HuggingFaceTB/SmolLM2-135M-Instruct
```

Dataset mặc định:

```text
Anthropic/hh-rlhf
```

Lý do chọn model nhỏ:

- Phù hợp laptop / GPU sinh viên
- Train nhanh hơn
- Ít VRAM hơn
- Dễ debug và giải thích thuật toán

Các stage vẫn giới hạn sample để phù hợp laptop, nhưng đã tăng nhẹ so với bản demo ban đầu:

- Mini SFT: `max_train_samples = 60`
- DPO: `max_train_samples = 100`
- PPO: `max_train_samples = 80`, `ppo_epochs = 3`

Các giá trị này giúp model có thêm cơ hội tạo khác biệt, nhưng vẫn không phải training quy mô paper.
Riêng PPO còn trộn thêm các prompt trong `evaluation/ood_cases.json` vào rollout để reward-based training nhìn thấy các tiêu chí mà benchmark sẽ chấm, ví dụ JSON format, code, safety, concise và no repetition.

## 5. Cài đặt

Tạo virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Cài thư viện:

```powershell
pip install -r requirements.txt
```

Kiểm tra GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## 6. Cách chạy khuyến nghị

Chạy menu chính:

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

Nguyên tắc quan trọng:

```text
Train once, evaluate many times.
```

Sau khi đã có model trong `outputs/`, thường chỉ cần dùng:

- `1. Compare models`
- `2. Plot toy probability heatmaps`
- `3. Run OOD benchmark` để lấy bảng điểm so sánh rõ hơn

Không chọn `9. Train all, then compare + benchmark` nếu không thật sự muốn train lại.

## 7. Chạy từng lệnh riêng

Train mini SFT:

```powershell
python train/sft.py
```

Train Reward Model:

```powershell
python train/reward_model.py
```

Train DPO:

```powershell
python train/dpo.py
```

Train PPO:

```powershell
python train/ppo.py
```

Compare model:

```powershell
python evaluation/compare.py --prompt "Human: Explain what RLHF is.\nAssistant:"
```

Nếu chỉ chạy:

```powershell
python evaluation/compare.py
```

chương trình sẽ hỏi prompt để nhập trực tiếp.

Vẽ toy probability heatmap ổn định cho báo cáo:

```powershell
python evaluation/plot_toy_probabilities.py --num-examples 8 --output outputs/probability_heatmaps.png --output-json outputs/probability_heatmaps_summary.json
```

Chạy sanity tests:

```powershell
python -m unittest tests/test_sanity.py
```

Chạy OOD benchmark:

```powershell
python evaluation/ood_benchmark.py --output-json outputs/ood_benchmark_result.json --output-html outputs/ood_benchmark_report.html
```

## 8. Output

Các output chính:

```text
outputs/
├── sft_model/
├── reward_model/
├── dpo_model/
├── ppo_model/
├── ppo_value_model/
├── sft_metrics.txt
├── reward_model_metrics.txt
├── dpo_metrics.txt
├── ppo_metrics.txt
├── compare_result.txt
├── compare_report.html
├── ood_benchmark_result.json
├── ood_benchmark_report.html
├── probability_heatmaps.png
├── probability_heatmaps_summary.json
├── probability_heatmaps_real.png
└── probability_heatmaps_real_summary.json
```

Ý nghĩa:

- `outputs/sft_model`: checkpoint sau mini SFT
- `outputs/reward_model`: checkpoint mô hình phần thưởng (Reward Model)
- `outputs/dpo_model`: policy model sau DPO
- `outputs/ppo_model`: policy model sau PPO (tối ưu hóa theo Hybrid Reward)
- `outputs/ppo_value_model`: value model của PPO
- `outputs/*_metrics.txt`: summary train cuối cùng, thường bị ghi đè khi train lại
- `outputs/compare_result.txt`: compare dạng text
- `outputs/compare_report.html`: compare dạng HTML 3 cột
- `outputs/ood_benchmark_result.json`: benchmark scores dạng JSON
- `outputs/ood_benchmark_report.html`: benchmark scores + responses dạng HTML
- `outputs/probability_heatmaps.png`: hình toy heatmap paper-style cho báo cáo
- `outputs/probability_heatmaps_summary.json`: summary cho toy heatmap, cho thấy DPO target probability giảm ở OOD còn PPO giữ cao hơn
- `outputs/probability_heatmaps_real.png`: xác suất chéo prompt-to-response thực tế sau hiệu chỉnh (calibrated) và co giãn nhiệt độ

## 9. Ghi đè hay tạo mới?

Hiện tại project dùng path cố định. Khi train lại, các file/thư mục sau thường bị **ghi đè**:

```text
outputs/sft_model
outputs/reward_model
outputs/dpo_model
outputs/ppo_model
outputs/ppo_value_model
outputs/sft_metrics.txt
outputs/reward_model_metrics.txt
outputs/dpo_metrics.txt
outputs/ppo_metrics.txt
```

Compare cũng ghi đè:

```text
outputs/compare_result.txt
outputs/compare_report.html
```

Vì vậy tắt máy vẫn còn checkpoint, miễn là không xóa thư mục `outputs/`.

## 10. Cách diễn giải kết quả

Không nên chỉ nhìn một response rồi kết luận model nào tốt hơn. Với demo này nên dùng các lớp đánh giá:

1. **Compare report**: xem trực quan Base / DPO / PPO trả lời khác nhau thế nào.
2. **Toy probability heatmap**: minh họa paper-style về data coverage của DPO và reward-guided behavior của PPO.
3. **OOD benchmark**: dùng nhiều prompt stress test và scoring rule-based, nên phù hợp nhất để trình bày kết quả demo.

Khi trình bày:

- Base model là baseline.
- DPO học trực tiếp từ preference pairs.
- PPO học qua reward signal và KL penalty.
- Kết quả chỉ minh họa cơ chế vì model nhỏ, sample ít, reward học được kết hợp rule-based.

## 11. Liên hệ với paper

Paper nghiên cứu ở quy mô lớn hơn nhiều, với training tối ưu hơn và benchmark nghiêm túc hơn. Project này chỉ lấy cảm hứng từ các ý chính:

- So sánh DPO và PPO trong alignment.
- Phân tích preference data và learned policy.
- Quan sát rằng DPO có thể bị giới hạn bởi dữ liệu preference đã có.
- Quan sát rằng PPO có thêm cơ chế reward-based optimization, nhưng cũng phức tạp và nhạy hơn.

Khi báo cáo, nên nói:

> Demo không chứng minh PPO luôn tốt hơn DPO. Demo chỉ minh họa rằng DPO và PPO tối ưu theo hai cơ chế khác nhau, và trong bối cảnh dữ liệu lạ hoặc tiêu chí reward cụ thể, PPO có thể có lợi thế nếu reward signal được thiết kế tốt.

## 12. Hạn chế

- Mini SFT dùng ít sample.
- DPO/PPO train local, batch nhỏ.
- PPO dùng rule-based reward, không phải learned reward model.
- Không benchmark nhiều seed.
- Không có đánh giá người dùng thật.
- Toy probability heatmap là công cụ minh họa, không phải kết quả paper-level.

## 13. Kết luận ngắn

Project phù hợp để báo cáo môn học vì cho thấy rõ:

- Preference dataset hoạt động như thế nào
- SFT, DPO, PPO khác nhau ở đâu
- Policy/reference model dùng để làm gì
- Log-probability, KL, reward, advantage xuất hiện trong code ra sao
- Vì sao không nên đánh giá alignment chỉ bằng một response đơn lẻ
