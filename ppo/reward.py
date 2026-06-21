"""
reward.py

Reward function cho PPO training.

Mục tiêu:
- đơn giản
- ổn định
- dễ debug
"""

import re


# ==========================================================
# 1. Reward cơ bản theo rule
# ==========================================================

def rule_based_reward(response: str) -> float:
    """
    Chấm điểm câu trả lời theo rule đơn giản.

    Ý tưởng:
    - Có câu trả lời → + điểm
    - Quá ngắn → phạt
    - Có dấu hiệu lạc đề → phạt
    """

    if response is None:
        return -1.0

    response = response.strip()

    # --------------------------
    # quá ngắn
    # --------------------------
    if len(response) < 10:
        return -0.5

    # --------------------------
    # quá dài (hơi phạt nhẹ)
    # --------------------------
    if len(response) > 500:
        return -0.1

    # --------------------------
    # check có nội dung cơ bản
    # --------------------------
    has_letters = re.search(r"[a-zA-ZÀ-ỹ]", response)
    if not has_letters:
        return -0.5

    # --------------------------
    # reward cơ bản
    # --------------------------
    return 1.0


# ==========================================================
# 2. Reward chính dùng cho PPO
# ==========================================================

def compute_reward(query: str, response: str) -> float:
    """
    Reward function chính.

    Có thể mở rộng sau:
    - sentiment model
    - reward model
    - safety filter
    """

    base_reward = rule_based_reward(response)

    # ------------------------------------------------------
    # bonus nếu response có liên quan query
    # ------------------------------------------------------
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())

    overlap = len(query_words.intersection(response_words))

    relevance_bonus = min(overlap * 0.1, 0.5)

    # ------------------------------------------------------
    # final reward
    # ------------------------------------------------------
    reward = base_reward + relevance_bonus

    # clamp reward để tránh PPO explosion
    reward = max(-1.0, min(2.0, reward))

    return reward


# ==========================================================
# 3. Test nhanh
# ==========================================================

if __name__ == "__main__":

    q = "What is Artificial Intelligence?"

    r1 = "AI is a field of computer science that builds intelligent systems."
    r2 = "ok"
    r3 = "Artificial Intelligence is used in machine learning, robotics, and NLP."

    print("Reward 1:", compute_reward(q, r1))
    print("Reward 2:", compute_reward(q, r2))
    print("Reward 3:", compute_reward(q, r3))