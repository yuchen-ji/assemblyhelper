import os
import sys
import yaml

fname = os.path.join("/home/yuchen/projects/HRCA", "experiments/fewshot/results_gpt35_cot_3shot_sot.yml")
dname = os.path.join("/home/yuchen/projects/HRCA", "datasets/dataset_all.yml")
oname = os.path.join("/home/yuchen/projects/HRCA", "experiments/fewshot/results_gpt35_cot_3shot_sot_.yml")

with open(fname, "r") as f:
    results = f.read()

with open(dname, "r") as f:
    dataset = f.read()

result_list = results.split("\n\n")
dataset_list = dataset.split("\n\n")

raw_dataset = ""
for idx, data in enumerate(dataset_list):
    if not data.startswith("# "):
        continue
    if data.startswith("# 任务"):
        continue
    raw_dataset += data + "\n\n"
    # print(data)
    # print("================================================")
print(raw_dataset)

raw_result = ""
for idx, result in enumerate(result_list):
    if not result.startswith("# "):
        continue
    raw_result += result + "\n\n"
    # print(result)
    # print("================================================")
print(raw_result)

raw_dataset = raw_dataset.split("\n\n")
raw_result = raw_result.split("\n\n")

print(len(raw_dataset))
print(len(raw_result))

final_result = ""
for raw_data, raw_res in zip(raw_dataset, raw_result):
    data_no = raw_data.split("\n")[0]
    res_no = raw_res.split("\n")[0]
    # print(data_no)
    # print(res_no)
    final_res = data_no + "\n" + "\n".join(raw_res.split("\n")[1:])
    final_result += final_res + "\n\n"

print(final_result)


with open(oname, "w") as f:
    f.write(final_result)
