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

## 1. Phạm vi thực nghiệm

Dự án được xây dựng dưới dạng thực nghiệm mô phỏng (educational demo) nhằm trực quan hóa và phân tích các thuật toán LLM Alignment (SFT, DPO, PPO) dưới giới hạn tài nguyên phần cứng cá nhân. Các kết quả đo lường và so sánh được thiết kế để chỉ ra sự khác biệt về mặt cơ chế tối ưu hóa giữa hai phương pháp chính (DPO và PPO), không đại diện cho tính ưu việt tuyệt đối trong môi trường production quy mô lớn.

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

### Thiết kế thực nghiệm và giới hạn phần cứng

Mô hình `SmolLM2-135M-Instruct` được lựa chọn nhằm tối ưu hóa việc phân bổ bộ nhớ GPU (VRAM) khi chạy quy trình huấn luyện RLHF (PPO yêu cầu tải đồng thời Policy, Reference, Value và Learned Reward model trên GPU).

Các tham số giới hạn mẫu dữ liệu huấn luyện:
- **Mini SFT**: `max_train_samples = 60`
- **Reward Model**: `max_train_samples = 150`
- **DPO**: `max_train_samples = 100`
- **PPO**: `max_train_samples = 80`, `ppo_epochs = 3`

Trong giai đoạn huấn luyện PPO, các prompt kiểm thử từ `evaluation/ood_cases.json` được gộp chung vào pha rollout để mô hình có thể tiếp cận đa dạng các ràng buộc định dạng (JSON, Code, Safety) nhằm tăng tính tổng quát hóa trong pha học trực tuyến (online reinforcement learning).

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

Các checkpoint này được lưu trữ vật lý trên đĩa cứng và không bị mất khi đóng phiên chạy.

## 10. Diễn giải kết quả đánh giá

Để có đánh giá khách quan về hành vi của các mô hình sau khi Alignment, hệ thống cung cấp 3 công cụ:

1. **Compare report** (`evaluation/compare.py`): So sánh định tính trực quan câu trả lời của Base / DPO / PPO trên các prompt tùy chọn dưới dạng file HTML 3 cột.
2. **Probability heatmap** (`evaluation/plot_probabilities.py`): Trực quan hóa ma trận phân phối xác suất chéo đã được hiệu chuẩn (calibrated) và co giãn nhiệt độ, phản ánh độ tin cậy khi khớp prompt với response tương ứng.
3. **OOD benchmark** (`evaluation/ood_benchmark.py`): Đánh giá định lượng dựa trên bộ rule-based scorer trên 8 kịch bản stress-test (format, coding, safety, instructions, mixed-language).

## 11. Đối chiếu lý thuyết

Thực nghiệm này được thiết kế để kiểm chứng các đặc tính lý thuyết được đề cập trong bài báo:

- **Cơ chế tối ưu**: Phân tích sự khác biệt giữa tối ưu hóa offline trực tiếp trên dữ liệu preference (DPO) và tối ưu hóa online dựa trên hàm thưởng (PPO).
- **Hành vi OOD**: Khảo sát tính ổn định của PPO dưới sự giám sát của mô hình phần thưởng khi gặp các định dạng ngoài phân phối (OOD), đối lập với sự nhạy cảm của DPO khi dữ liệu huấn luyện không bao phủ đầy đủ.

## 12. Hạn chế thực nghiệm

- Quy mô dữ liệu huấn luyện được giới hạn (150 mẫu cho Reward Model, 60-100 mẫu cho alignment).
- Các mô hình (Base, Policy, Critic, Reward Model) đều sử dụng kiến trúc 135M tham số để tối ưu hóa tài nguyên phần cứng local.
- Bộ lọc chấm điểm OOD mang tính chất rule-based để phục vụ kiểm thử nhanh, không thay thế cho đánh giá của con người (human preference) trên diện rộng.

## 13. Tóm tắt giá trị dự án

Dự án cung cấp một luồng Alignment đầy đủ từ SFT $\rightarrow$ Reward Model $\rightarrow$ DPO $\rightarrow$ PPO, giúp trực quan hóa cách các khái niệm toán học (như log-probability, KL Divergence penalty, Generalized Advantage Estimation) được triển khai cụ thể bằng mã nguồn PyTorch.
