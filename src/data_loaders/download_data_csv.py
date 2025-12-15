from datasets import load_dataset

ds = load_dataset("google/code_x_glue_cc_clone_detection_big_clone_bench")

ds_train = ds['train']
df_train = ds_train.to_pandas()

df_train.to_csv('data/df_train.csv', index=False)
print("df_train.csv saved successfully.")

ds_test = ds['test']
df_test = ds_test.to_pandas()

df_test.to_csv('data/df_test.csv', index=False)
print("df_test.csv saved successfully.")

ds_val = ds['validation']
df_val = ds_val.to_pandas()

df_val.to_csv('data/df_val.csv', index=False)
print("df_val.csv saved successfully.")