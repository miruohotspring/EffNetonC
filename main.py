from efficientnet_pytorch import EfficientNet
import numpy as np

def main():
    np.set_printoptions(suppress=True)
    path = "./data/"
    model = EfficientNet.from_pretrained("efficientnet-b0")
    module_list = ""
    
    for m in model.state_dict():
        #メタデータの読み込み
        size = model.state_dict()[m].size()
        data = str(len(size))
        for s in size:
            data += "," + str(s)
        
        #データの読み込み
        weight = model.state_dict()[m].view(-1).numpy()
        for w in weight:
            data += "," + str(w)
        
        #書き出し
        module_list += (m + "\n");
        with open(path + m + ".txt", 'w') as f:
            print(data, file=f)
        
    
    #with open(path + "module_list.txt", 'w') as f:
    #    print(module_list, file=f)


if __name__ == "__main__":
    main()
    