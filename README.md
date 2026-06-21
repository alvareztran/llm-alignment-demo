# Alignment Demo: DPO và PPO cho LLM Alignment

Project này là một demo nhỏ về **alignment cho Large Language Model (LLM)**, tập trung vào hai hướng tối ưu preference phổ biến:

- **DPO (Direct Preference Optimization)**
- **PPO (Proximal Policy Optimization) trong RLHF**

Mục tiêu chính của project là phục vụ học tập và báo cáo môn Machine Learning / NLP: nhìn thấy dữ liệu đi qua tokenizer, model, log-probability, loss và bước update. Code được viết trực tiếp bằng PyTorch/Transformers thay vì dùng `trl.DPOTrainer` hoặc `trl.PPOTrainer` để dễ đọc và dễ giải thích.

## 1. Phạm vi của project

Đây là **educational implementation**, không phải hệ thống RLHF production-grade.

Project làm được:

- Chuẩn bị dữ liệu preference từ `Anthropic/hh-rlhf`
- Train DPO với policy model và frozen reference model
- Train một bản PPO actor-critic rút gọn
- Có reward signal, value function, GAE, KL penalty và clipped PPO objective
- So sánh output giữa base model, DPO model và PPO model

Project cố ý đơn giản hóa:

- Không có SFT stage thật; project bắt đầu từ một instruct model có sẵn
- PPO dùng reward rule-based, không train reward model từ preference data
- PPO chạy sample-by-sample, batch rất nhỏ
- Chưa có value clipping, entropy bonus, reward whitening, minibatch PPO đầy đủ
- Chưa có EMA update cho reference model như một số implementation PPO mạnh

Vì vậy, khi trình bày báo cáo nên nói:

> DPO trong project gần với objective gốc của paper. PPO trong project là bản demo rút gọn của PPO-RLHF, dùng để minh họa pipeline actor-critic và các thành phần chính.

## 2. Kiến trúc tổng quan

```text
alignment-demo/
├── configs/
│   ├── dpo_config.py
│   └── ppo_config.py
├── data/
│   └── prepare_dataset.py
├── dpo/
│   └── train_dpo.py
├── ppo/
│   ├── __init__.py
│   ├── reward.py
│   └── train_ppo.py
├── train/
│   ├── dpo.py
│   ├── ppo.py
│   └── sft.py
├── models/
│   ├── load_model.py
│   ├── policy_reference.py
│   ├── reward_model.py
│   └── value_model.py
├── utils/
│   ├── logprob.py
│   ├── advantage.py
│   └── sampling.py
├── evaluation/
│   ├── generate.py
│   └── compare.py
├── outputs/
├── run.py
└── requirements.txt
```

## 3. Các thành phần chính

| File | Vai trò |
|---|---|
| `data/prepare_dataset.py` | Load `Anthropic/hh-rlhf`, tách dữ liệu thành `prompt`, `chosen`, `rejected` |
| `models/load_model.py` | Load tokenizer và causal LM mặc định |
| `models/policy_reference.py` | Load policy model và frozen reference model |
| `utils/logprob.py` | Tính token log-prob bằng shifted logits cho causal LM |
| `train/dpo.py` | Train DPO bằng preference pairs |
| `ppo/reward.py` | Reward rule-based cho PPO demo |
| `models/reward_model.py` | Wrapper reward model cho PPO |
| `models/value_model.py` | Critic/value function token-level cho PPO |
| `utils/advantage.py` | Tính GAE và normalize advantage |
| `train/ppo.py` | Train PPO actor-critic rút gọn |
| `evaluation/compare.py` | So sánh base, DPO và PPO model |
| `run.py` | Menu chạy DPO, PPO, compare hoặc toàn bộ pipeline |

## 4. Dữ liệu

Dataset mặc định:

```text
Anthropic/hh-rlhf
```

Mỗi sample gốc có hai response:

- `chosen`: response được đánh giá tốt hơn
- `rejected`: response bị đánh giá kém hơn

File `data/prepare_dataset.py` tách phần hội thoại thành:

- `prompt`
- `chosen`
- `rejected`

Mặc định project chỉ lấy một số lượng nhỏ sample để train nhanh trên máy cá nhân. Có thể chỉnh trong:

- `configs/dpo_config.py`
- `configs/ppo_config.py`

## 5. DPO trong project

DPO không cần reward model riêng. Nó tối ưu trực tiếp policy từ dữ liệu preference.

Trong `train/dpo.py`, loss được triển khai theo dạng:

```text
-log sigmoid(beta * ((log pi(chosen) - log pi(rejected))
                   - (log ref(chosen) - log ref(rejected))))
```

Các thành phần:

- **Policy model**: model đang được fine-tune
- **Reference model**: bản frozen của base model, dùng để regularize policy
- **Chosen/rejected log-prob**: tổng log-prob trên các token response
- **Beta**: hệ số điều chỉnh độ mạnh của preference objective

Đánh giá: phần DPO này **gần với objective chuẩn của paper**, nhưng vẫn là bản nhỏ: không batch lớn, không distributed training, không dùng trainer của TRL.

## 6. PPO trong project

PPO trong project mô phỏng pipeline RLHF:

1. Policy model sinh response cho prompt
2. Reward model rule-based chấm điểm response
3. Reference model tính log-prob để tạo KL penalty
4. Value model ước lượng value cho từng token
5. GAE tính advantage và return
6. PPO update policy bằng clipped surrogate objective

Các công thức chính có trong code:

```text
ratio = exp(new_logprob - old_logprob)
policy_loss = -mean(min(ratio * advantage,
                        clip(ratio, 1-eps, 1+eps) * advantage))
value_loss = MSE(value, return)
reward = score_reward - kl_coef * (log pi_old - log pi_ref)
```

Đánh giá: phần PPO này **đúng ý tưởng actor-critic và có các thành phần quan trọng**, nhưng là **simplified PPO-RLHF demo**, không phải PPO đầy đủ như các framework lớn.

Những điểm đã có:

- Policy model
- Frozen reference model
- Rule-based reward model
- Value function
- Old log-prob và new log-prob
- Clipped objective
- KL penalty
- GAE
- Advantage normalization

Những điểm chưa có so với implementation mạnh hơn:

- Learned reward model từ preference pairs
- Batch/minibatch PPO đúng nghĩa
- Value loss clipping
- Reward normalization/whitening
- Entropy bonus
- Reference model EMA
- Scale training lớn như DeepSpeed-Chat, TRL hoặc ReaLHF

## 7. SFT trong project

File `train/sft.py` hiện chỉ là placeholder.

Project không train SFT từ đầu vì model mặc định đã là instruct model:

```text
HuggingFaceTB/SmolLM-135M-Instruct
```

Khi báo cáo, nên nói rõ:

> Project bỏ qua SFT stage thật để giảm thời gian và tài nguyên train. Base model instruct được dùng như điểm khởi đầu thay cho SFT model.

Nếu muốn làm pipeline RLHF đầy đủ hơn, có thể thêm SFT bằng supervised cross-entropy trên dữ liệu demonstration trước khi chạy DPO/PPO.

## 8. Model mặc định

Model mặc định nằm trong `models/load_model.py`:

```python
MODEL_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"
```

Lý do chọn model nhỏ:

- Dễ chạy trên máy cá nhân
- Ít VRAM hơn
- Thời gian train ngắn hơn
- Phù hợp để debug và giải thích thuật toán

Có thể đổi sang model khác, ví dụ:

```python
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
```

Khi đổi model, cần chú ý VRAM, tokenizer và tốc độ train.

## 9. Cài đặt

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

Dataset được tải từ Hugging Face, nên lần chạy đầu cần internet.

## 10. Cách chạy

Train DPO:

```powershell
python train/dpo.py
```

hoặc:

```powershell
python dpo/train_dpo.py
```

Train PPO:

```powershell
python train/ppo.py
```

hoặc:

```powershell
python ppo/train_ppo.py
```

So sánh model:

```powershell
python evaluation/compare.py
```

Chạy bằng menu:

```powershell
python run.py
```

Menu gồm:

```text
1. Train DPO
2. Train PPO
3. Compare models
4. Run all
```

## 11. Output

Các output chính:

```text
outputs/
├── dpo_model/
├── ppo_model/
├── ppo_value_model/
├── dpo_metrics.txt
├── ppo_metrics.txt
└── compare_result.txt
```

Ý nghĩa:

- `outputs/dpo_model`: policy model sau DPO
- `outputs/ppo_model`: policy model sau PPO
- `outputs/ppo_value_model`: value model của PPO
- `outputs/dpo_metrics.txt`: loss và thông tin train DPO
- `outputs/ppo_metrics.txt`: reward, KL, policy loss, value loss của PPO
- `outputs/compare_result.txt`: output so sánh base/DPO/PPO

## 12. Cách diễn giải kết quả

Base model:

- Là model gốc trước khi train thêm trong project
- Dùng làm baseline

DPO model:

- Học trực tiếp từ `chosen` và `rejected`
- Không cần reward model riêng
- Phù hợp để minh họa preference optimization

PPO model:

- Học từ reward signal sinh ra trong quá trình rollout
- Có KL penalty để hạn chế lệch quá xa reference model
- Kết quả phụ thuộc rất nhiều vào reward function
- Vì reward hiện là rule-based nên kết quả chỉ phản ánh tiêu chí đơn giản trong `ppo/reward.py`

Không nên kết luận từ project này rằng DPO luôn hơn PPO hoặc PPO luôn hơn DPO. Project chỉ dùng model nhỏ, sample ít và PPO rút gọn, nên kết quả chủ yếu dùng để minh họa cơ chế.

## 13. Đối chiếu với paper

So với paper và các hệ thống RLHF đầy đủ:

Phần gần chuẩn:

- DPO có policy/reference model và log-ratio preference objective
- PPO có actor, critic, rollout, KL penalty, GAE và clipped policy objective

Phần demo/rút gọn:

- Không train SFT thật
- Không train reward model từ preference data cho PPO
- Batch size nhỏ
- Không có đầy đủ các trick ổn định hóa PPO quy mô lớn
- Không benchmark nhiều task hoặc nhiều seed

Cách viết phù hợp trong báo cáo:

> Project triển khai lại các thành phần cốt lõi của DPO và PPO-RLHF ở quy mô nhỏ. DPO gần với công thức gốc. PPO giữ actor-critic, clipped objective, value function, GAE và KL penalty, nhưng reward model và training loop được đơn giản hóa để phù hợp mục tiêu học tập.

## 14. Khi nào nên dùng TRL Trainer?

Không bắt buộc dùng `trl.DPOTrainer` hoặc `trl.PPOTrainer` cho báo cáo môn học.

Nên giữ code tự viết nếu mục tiêu là:

- Giải thích công thức
- Chỉ ra loss nằm ở đâu
- Hiểu log-prob, advantage, KL và value function
- Bảo vệ project trước giảng viên

Nên dùng TRL nếu mục tiêu là:

- Train nghiêm túc hơn
- Dùng batch/minibatch, logging, checkpoint tốt hơn
- Giảm rủi ro bug implementation
- So sánh kết quả gần với thực nghiệm chuẩn hơn

Một hướng tốt cho báo cáo là giữ implementation hiện tại, rồi ghi thêm rằng TRL/DeepSpeed-Chat/ReaLHF là lựa chọn phù hợp hơn khi scale lên.

## 15. Kết luận

Project này phù hợp để học và trình bày các khái niệm chính của LLM alignment:

1. Preference dataset
2. Policy model và reference model
3. DPO loss
4. Reward signal trong PPO
5. Value function và advantage estimation
6. KL regularization
7. So sánh output trước và sau alignment

Kết luận ngắn gọn:

- **DPO**: triển khai gần với objective paper, phù hợp để báo cáo.
- **PPO**: triển khai đúng ý tưởng chính nhưng là bản demo đơn giản hóa.
- **SFT**: chưa train thật, chỉ dùng instruct model có sẵn làm điểm khởi đầu.
