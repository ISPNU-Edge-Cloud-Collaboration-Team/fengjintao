用GNN训练延迟预测器

11.9 

数据集采集

论文：[[2007.08668\] BRP-NAS: Prediction-based NAS using GCNs (arxiv.org)](https://arxiv.org/abs/2007.08668)

1.用cifar10数据集训练一个超网络mysupernet.pt

​           正确率：92.76 推理时间：20.395668268203735

2.对超网络进行剪枝

剪枝主要代码：

```
for i in range(1, 120):
    arch=[]
    a=4
    count=0
    arch_code = getSuperCode(10)
    while count <= a:
        arch_code = getSuperCode(10)
        arch_code,result = prune_arch_code(arch_code, i)
        count=count+1
        if result:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            MYmodel = torch.load("/home/data/hw/fjt/DGNAS/modelcut/mysupernet.pt",
                                 map_location=device)
            save_arch_code(arch_code)
            MYmodel.setArch(arch_code)
            print("成功剪枝", global_count)
            save_path = f"/home/data/hw/fjt/DGNAS/modelcut/MYmodel_{global_count}.pt"  # 使用计数器编码模型文件名
            global_count += 1
            torch.save(MYmodel, save_path)
        else:
            # 剪枝操作失败
            # 处理剪枝失败的情况
            print("剪枝操作失败，退出循环")
            break  # 跳出循环
    print(i)
```

其中记录每一个剪枝模型的arch_code作为模型结构，保存剪枝完成的pt文件，随机剪出450个pt模型  先转化为onnx 文件再转为mnn文件 在自己的电脑上跑每个模型100次取平均时间作为延迟数据标签。

转化为onnx的代码:

```
for i in range(595):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/data/hw/fjt/DGNAS/modelcut/MYmodel_{}.pt".format(counter)
    model = torch.load(model_path, map_location=device)
    onnx_path = "/home/data/hw/fjt/DGNAS/myonnx/MYonnxmodel_{}.onnx".format(counter)
    print(device)
    torchToOnnx(model,onnx_path, output_names=['output'],dyn=None)
    counter += 1
    print(f"步数: {counter}")
```

转为MNN的代码：

```
onnx_folder = "/home/data/hw/fjt/DGNAS/myonnx/"
mnn_folder = "/home/data/hw/fjt/DGNAS/mymnn/"
for file_name in os.listdir(onnx_folder):
  if file_name.endswith(".onnx"):
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     onnxname = os.path.join(onnx_folder, file_name)
     #model = torch.load(model_path, map_location=device)
     mnnname = os.path.join(mnn_folder, file_name.replace(".onnx", ".mnn"))
     onnxToMnn(onnxname,mnnname, path="/home/data/hw/Z_bing333/package/MNN/build/")
```



3.通过archCodeEncoder将保存的arch_codes中的模型结构转化为GNN可以读取的输入张量

```
def archCodeEncoder(archCode,layer_num):
    edge_index, edge_fea=[[],[]],[]
    layerNodeCnt=6
    n=layer_num * layerNodeCnt
    adList,_=archCodeAdList(archCode,layer_num)
    node_fea=torch.ones((n,1))
    for i in range(n):
        for edge in adList[i]:
            edge_index[0].append(edge[0])
            edge_index[1].append(edge[1])
            ze=torch.zeros((4))
            ze[edge[2]]=1
            edge_fea.append(ze)
    edge_fea=torch.stack(edge_fea,dim=0)
    # print(edge_fea.shape)
    return node_fea, torch.tensor(edge_index,dtype=torch.long), edge_fea
```







