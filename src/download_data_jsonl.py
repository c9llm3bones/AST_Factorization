from datasets import load_dataset
import json

ds = load_dataset("google/code_x_glue_cc_clone_detection_big_clone_bench")

def save_jsonl(split_name, filename):
    df = ds[split_name].to_pandas()
    with open(filename, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            obj = {
                "pair_id": str(row['id']),
                "id1": str(row['id1']),
                "id2": str(row['id2']),
                "label": int(row['label']),
                "func1": row['func1'],
                "func2": row['func2']
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"{filename} saved successfully.")

save_jsonl('train', 'data/train.jsonl')
print('Succesfully saved train')
save_jsonl('test', 'data/test.jsonl')
print('Succesfully saved test')
save_jsonl('validation', 'data/val.jsonl')
print('Succesfully saved val')