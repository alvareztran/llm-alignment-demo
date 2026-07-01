import json

with open("./outputs/ood_benchmark_result.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("SUMMARY BY MODEL:")
print(json.dumps(data["summary"]["by_model"], indent=2))
print("\nSUMMARY BY CATEGORY:")
print(json.dumps(data["summary"]["by_category"], indent=2))

print("\nRESPONSES ON SAFETY & FORMAT:")
results = data["results"]
for r in results:
    if r["case_id"] in ["safety_password", "json_api"]:
        print(f"\nModel: {r['model'].upper()} | Case: {r['case_id']}")
        print(f"Prompt: {r['prompt'].strip()}")
        print(f"Response: {r['response'].strip()}")
        print(f"Score Total: {r['score']['total']}")
        print("-" * 50)
