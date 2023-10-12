# **AssemblyHelper:** Foundation Model for Human-Robot Collaboration Assembly

## **实验环境**
- FR3协作机器人
- 人类协作者
- 卫星组件(框架x1, 电池x1, 桁条x4, 信息转接板x1)
- 紧固件(十字螺钉, 一字螺钉, 内六角螺钉)
- 工具(十字螺丝刀, 一字螺丝刀, 内六角螺丝刀)
- 工作空间(装配空间, 零件空间, 工具空间, 交换空间)


## **装配流程 (human & robot)**
**一、安装桁条1**
| step | Human | Robot |
| :---- | :---- | :---- |
| 1 | [action] 我拿起了内六角螺钉 | [code] 将内六角螺丝刀移动到交换空间
| 2 | [action] 我伸右手去接取物品 | [code] 松开夹爪
| 3 | [language] 将夹爪移动到装配空间 | [code] follow human's instruction
| 4 | [action&language] 将桁条移动到我手指的位置 | [img&code] 将桁条移动到左上角的安装位置
| 4 | (alternative)[language] 将桁条移动到左上角 | [code] follow human's instruction
| 5 | [action] 装配桁条 | [code] No action
| 6 | [action] stand | [code] No action

**二、安装桁条2**
| step | Human | Robot |
| :---- | :---- | :---- |
| 1 | [language] 将桁条递给我 | [code] follow human's instruction
| 2 | [action] 伸右手去拿物体 | [code] 松开夹爪
| 3 | [action] 我拿起了内六角螺钉 | [code] No action, 因为内六角螺丝刀已经在人手中
| 4 | [language] 装配桁条 | [code] No action
| 5 | [language] stand | [code] No action

**三、安装电池左侧**
| step | Human | Robot |
| :---- | :---- | :---- |
| 1 | [language] 将电池移动到装配的位置 | [code] follow human's instruction
| 2 | [action] 拿起内六角螺钉 | [code] No action, 因为内六角螺丝刀在手中
| 3 | [action] 装配电池 | [code] No action
| 4 | ....
| 5 | [action] stand | [code] No action

**四、安装桁条3**
| step | Human | Robot |
| :---- | :---- | :---- |
| 1 | [language] 将夹爪移动到装配空间 | [code] follow human's instruction
| 2 | [action&language] 将桁条移动到我手指的位置 | [img&code] 将桁条移动到右上角的装配处
| 2 | (alternative)[language] 将桁条移动到右上角 | [code] follow human's instruction
| 3 | [action] 我拿起了内六角螺钉 | [code] No action, 因为内六角螺丝刀已经在人手中
| 4 | [action] 装配桁条 | [code] No action
| 5 | [action] stand | [code] No action

**五、安装桁条4**
| step | Human | Robot |
| :---- | :---- | :---- |
| 1 | [language] 将夹爪移动到装配空间 | [code] follow human's instruction
| 2 | [action&language] 桁条移动到我手指的位置 | [img&code] 将桁条移动到右下角的装配处
| 2 | (alternative)[language] 将桁条移动到右下角 | [code] follow human's instruction
| 3 | [action] 我拿起了内六角螺钉 | [code] No action, 因为内六角螺丝刀已经在人手中
| 4 | [action] 装配桁条 | [code] No action
| 5 | [action] stand | [code] No action

**六、安装信号转接板**
| step | Human | Robot |
| :---- | :---- | :---- |
| 1 | [action] 我拿取了一字螺钉 | [code] 将一字螺丝刀移动到交换空间
| 2 | [action] 伸出右手拿东西 | [code] 松开夹具
| 3 | [action] stand | [code] No action
| 4 | [lanuage] 将信号转接板移动到待安装的位置 | [code] follow human's instruction
| 5 | [action] 装配信号转接板 | [code] No action
| 6 | [action] 我拿去了一字螺钉 | [code] No action, 因为一字螺丝刀已经在手中
| 7 | [action] 装配信号转接板 | [code] No action
| 8 | [action] stand | [code] No action


## **Related human action**
1. stand （still）
2. pick up fastener （extend left hand， dynamic）
3. receive tools or parts （extend right hand，still）
4. assembly （extend both hands， still）
5. other adversarial action

## **Memory bank**
**Robot state**
| step | location | status | object |
| :---- | :---- | :---- | :---- |
| 1 | deliver_space | closed | None |
| n | when use `move_to_location()` | when use `open()` or `close()` | when use `open()` set object = None, when using `close()` set object = pick_obj |


**Scene state (NOT NECESSARY)**
| step | part_space | tool_space | assembly_space |
| :---- | :---- | :---- | :---- |
| 1 | all parts | all tools | framework |
| n | ... | ... | ... | ... |


# TODO
- [ ] 固定相机的末端夹具的设计
- [ ] 购买一个有手柄的六角扳手
- [ ] LLM人手指向的物体识别
- [ ] 零件和工具数据集收集
- [ ] 零件识别，工具识别，装配体识别的验证及可视化
- [ ] 相机固定架的定做
- [ ] 周围环境的覆盖遮罩购买
- [ ] RealSense 接入以及手眼标定
- [ ] pixel2cam, cam2end, end2base 函数编写 
- [ ] robot_state_update 函数编写
- [ ] 行为识别数据集的收集
