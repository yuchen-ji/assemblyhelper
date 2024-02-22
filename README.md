# AssemblyHelper: Foundation Models Assist in Human–Robot Collaboration Assembly

## Human-Robot Collaboration Assembly Framework
### Overview
The architecture of the AssemblyHelper includes a perception module, a reasoning module, and an execution module. When performing manufacturing tasks, the workflow of the HRC system is as follow:

<!-- ![](/assets/figures/overview.svg#pic_center) -->
<div align="center">
    <img src="https://github.com/yuchen-ji/assemblyhelper/blob/main/assets/figures/overview.svg?sanitize=true" width="70%">
</div>

<!-- ### System implement
![](/assets/figures/system.svg#pic_center) -->


## Case study
### Experiment setup
![](/assets/figures/case_all.svg#pic_center)


### Assembly location recognition
![](/assets/figures/case_pointed.svg#pic_center)


### HRC code reasoning
![](/assets/figures/case_llm_all.svg#pic_center)


### Robot visual grasping
![](/assets/figures/case_grasp_parts.svg#pic_center)


### Procedure of HRC assembly
![](/assets/figures/case_stringer1.svg#pic_center)


## Experiments
### Task reasoning
**Datasets:** The HRCA-Code dataset for HRC task reasoning include
26 simple tasks, 24 medium tasks, and 16 difficult tasks. See [here](/experiments/llms/datasets) for more details. The HRCA-Env dataset for scene updating is a subset of HRCA-Code dataset, which includes (24-7) medium tasks and 16 hard tasks, excluding too simple tasks and tasks without any code.

**Results:** We test the HRC-prompt based LLMs performance on the HRC datasets. Exitensive experiments are conducted with distinct [`prompt strategies`](/experiments/llms/strategy), [`shot number`](/experiments/llms/fewshot), [`models`](/experiments/llms/model), [`human feedback`](/experiments/llms/model) on HRC task reasoning. For updating scene observations after executing HRC code, We also compare two methods, named [`union`](/experiments/llms/scene) and [`decomposition`](/experiments/llms/scene). Overall, the performance on the HRC task reasoning and scene updating can be seen in [here](/experiments/llms/best/gpt4_cot_3shot_sot_union_scene.pdf). For the satistical results on each experiment, it can be found in this [EXCEL](/experiments/experiments.xlsx). For more details about each task's response, please see each folder in [here](/experiments/llms) for experiments on different perspectives and the file end with `*.yml` describes the whole tasks (including task input & llm response) on each setting. Each file is named as `[models]\_[strategy_1]\_[strategy_n]\_[shot number]\_[scene generation method]`, such as `gpt4_cot_3shot_sot_union_scene`.

### Semantic segmentation
**Datasets:** The HRCA-Obj dataset for object classification, including 8 objects. The trainset with about 600 images per catergory. The testset with 40 images per catergory, including half of them occluded, can be accessed follow this [link](/experiments/vfms/datasets/testset). The label feature of trainset and PCA tranform matrix can be accessed in [here](/experiments/vfms/datasets/trainset).

**Results:** The results of VFMs for classification can be found in this [EXCEL](/experiments/experiments.xlsx). The evaluation code and image annotation tools placed in [here](/experiments/vfms/scripts).

### HRC assembly case
**Assets:** Some quantity results in HRC assembly is stored in [here](/assets/robots) and [here](/assets/vfms). We also provide a [notebook](/notebooks) for semantic segmentation in task processes. A webui for HRC task reasoning is in [here](/experiments/llms/scripts/scene_generator_webui.py).