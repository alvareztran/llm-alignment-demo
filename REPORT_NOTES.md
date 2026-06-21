# Report Notes: SFT, DPO, PPO Alignment Demo

File này tóm tắt cách trình bày project trong báo cáo môn học. Trọng tâm là phân biệt phần nào gần với paper và phần nào là demo đơn giản hóa.

## 1. Mục tiêu project

Project minh họa pipeline alignment cho LLM ở quy mô nhỏ:

1. Load preference dataset
2. Tách `prompt`, `chosen`, `rejected`
3. Train mini SFT trên response `chosen`
4. Train DPO trực tiếp từ preference pairs
5. Train PPO bằng rollout, reward, value model và policy update
6. So sánh response của base, DPO và PPO model

Điểm chính cần nhấn mạnh:

> Project ưu tiên tính dễ hiểu và khả năng giải thích thuật toán hơn là hiệu năng production.

## 2. Mini SFT: đúng ý tưởng, nhỏ về quy mô

Mini SFT trong project dùng supervised cross-entropy cho causal LM:

```text
input  = prompt + chosen_response
labels = -100 trên prompt tokens, token ids thật trên response tokens
loss   = cross-entropy chỉ trên response
```

Nên nói:

> SFT trong project là mini SFT để minh họa supervised fine-tuning. Nó đúng ý tưởng instruction tuning nhưng chỉ chạy ít sample.

Không nên nói:

> SFT này tương đương một SFT stage quy mô lớn trong RLHF production.

Lưu ý pipeline:

- Nếu `outputs/sft_model` tồn tại, DPO/PPO sẽ dùng checkpoint này làm base.
- Nếu chưa train SFT, DPO/PPO sẽ fallback về model gốc trong `models/load_model.py`.
- Bước compare mặc định chỉ hiển thị `base`, `dpo` và `ppo`; SFT là checkpoint trung gian, chưa được compare như một model riêng.
- Nếu muốn DPO/PPO chạy lại từ model gốc, cần đổi `base_model_path` hoặc xóa/đổi tên `outputs/sft_model`.

## 3. DPO: gần với paper

DPO trong project có đủ các thành phần cốt lõi:

- Policy model: model đang được update
- Reference model: frozen copy của base/SFT model
- Preference pair: `chosen` và `rejected`
- Log-prob response theo causal LM
- DPO objective dạng log-ratio

Công thức trong code:

```text
loss = -log sigmoid(beta * ((log pi(chosen) - log pi(rejected))
                          - (log ref(chosen) - log ref(rejected))))
```

Nên nói:

> Phần DPO gần với objective gốc, nhưng training loop được viết tối giản, chạy sample nhỏ và không dùng TRL Trainer.

Không nên nói:

> DPO implementation này tương đương các benchmark quy mô lớn.

## 4. PPO: demo actor-critic rút gọn

PPO trong project có các thành phần quan trọng:

- Policy model sinh response
- Frozen reference model để tính KL
- Rule-based reward model
- Value model làm critic
- Old/new log-prob
- GAE
- Clipped surrogate objective
- Value loss

Công thức chính:

```text
ratio = exp(new_logprob - old_logprob)
policy_loss = -mean(min(ratio * advantage,
                        clip(ratio, 1-eps, 1+eps) * advantage))
value_loss = MSE(value, return)
reward = score_reward - kl_coef * (log pi_old - log pi_ref)
```

Nên nói:

> PPO trong project đúng ý tưởng actor-critic và chứa các thành phần chính của PPO-RLHF, nhưng là bản demo rút gọn.

Không nên nói:

> PPO này là implementation đầy đủ như TRL, DeepSpeed-Chat hoặc ReaLHF.

## 5. Những phần đã đơn giản hóa

Các giới hạn cần nêu thẳng trong báo cáo:

- Mini SFT dùng ít sample, chưa phải SFT quy mô lớn
- PPO không train learned reward model từ preference data
- Reward hiện là rule-based
- Batch size nhỏ, loop sample-by-sample
- Không có value clipping
- Không có entropy bonus
- Không có reward whitening
- Không có reference EMA
- Không benchmark nhiều seed hoặc nhiều dataset

Các giới hạn này không làm project sai. Chúng chỉ xác định phạm vi:

> Đây là project minh họa thuật toán, không phải reproduction đầy đủ của paper.

## 6. Vì sao không dùng DPOTrainer/PPOTrainer?

Lý do hợp lý cho báo cáo:

- Code tự viết giúp nhìn rõ logprob, loss và gradient
- Dễ chỉ ra reference model, value function, KL penalty nằm ở đâu
- Phù hợp mục tiêu học thuật
- Giảm việc phụ thuộc vào logic ẩn trong thư viện

Câu trả lời khi bị hỏi:

> TRL Trainer phù hợp hơn khi scale training hoặc benchmark nghiêm túc. Project này viết trực tiếp bằng PyTorch/Transformers để giải thích thuật toán rõ hơn.

## 7. Checklist khi bảo vệ

Nếu giảng viên hỏi SFT có gì:

- Chỉ vào `train/sft.py`
- Nêu labels prompt bị mask bằng `-100`
- Nêu loss chỉ tính trên `chosen` response
- Nêu đây là mini SFT để minh họa, không phải SFT quy mô lớn

Nếu giảng viên hỏi DPO có đúng không:

- Chỉ vào `train/dpo.py`
- Nêu policy/reference log-ratio
- Nêu reference model frozen
- Nêu chosen/rejected đều được tính log-prob

Nếu giảng viên hỏi PPO có gì:

- Chỉ vào `train/ppo.py`
- Nêu rollout sinh response
- Nêu reward model rule-based
- Nêu KL penalty với reference model
- Nêu GAE trong `utils/advantage.py`
- Nêu value model trong `models/value_model.py`
- Nêu clipped objective trong `ppo_update`

Nếu giảng viên hỏi hạn chế:

- SFT mini, chưa quy mô lớn
- Reward model chưa learned
- PPO rút gọn để chạy nhanh
- Kết quả chỉ minh họa, không kết luận DPO luôn hơn PPO

## 8. Kết luận nên dùng

Kết luận phù hợp:

> Project cho thấy pipeline alignment gồm mini SFT, DPO và PPO ở quy mô nhỏ. DPO đơn giản hơn vì tối ưu trực tiếp từ preference pairs, còn PPO linh hoạt hơn vì tối ưu qua reward signal. Trong project này, DPO gần với công thức paper hơn, còn PPO được giữ ở mức actor-critic demo để minh họa reward-based alignment.
