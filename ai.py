import MeCab
import torch
import copy
import time
import matplotlib.pyplot as plt
import re
import math
import numpy as np
from gensim.models import Word2Vec
import pickle
import threading

class DenseBlock(torch.nn.Module):
    def __init__(self, dim, mul=1):
        super().__init__()
        self.I = torch.nn.Linear(dim, dim*mul)
        self.O = torch.nn.Linear(dim*mul, dim)
    def forward(self, x):
        x = self.I(x)
        x = torch.nn.functional.gelu(x)
        x = self.O(x)
        return x
class AttentionBlock(torch.nn.Module):
    def __init__(self, dim, mul=1):
        super().__init__()
        self.Q = torch.nn.Linear(dim, dim*mul)
        self.K = torch.nn.Linear(dim, dim*mul)
        self.V = torch.nn.Linear(dim, dim*mul)
        self.O = torch.nn.Linear(dim*mul, dim)
    def forward(self, q,k,v):
        q = self.Q(q)
        k = self.K(k)
        v = self.V(v)
        x = torch.nn.functional.softmax(q * k, dim=-1) * v
        x = self.O(x)
        return x

class SanokaLayer(torch.nn.Module):
    def __init__(self, dim, mul=1):
        super().__init__()
        self.x = None
        self.A = AttentionBlock(dim, mul)
        self.B = DenseBlock(dim, mul)
    def reset(self, x=None):
        self.x = x
    def forward(self, u):
        if (self.x != None):
            uu = torch.nn.functional.normalize(u)
            xx = torch.nn.functional.normalize(self.x)
            x = self.A(uu, xx, xx) + self.x
            y = self.B(torch.nn.functional.normalize(x)) + u
            self.x = x
            return y
        else:
            uu = torch.nn.functional.normalize(u)
            x = self.A(uu, uu, uu)
            y = self.B(torch.nn.functional.normalize(x)) + u
            self.x = x
            return y

class SanokaModel(torch.nn.Module):
    def __init__(self, dim, mul=1, Top=True):
        super().__init__()
        self.Top = Top
        if (Top):
            self.I = torch.nn.Linear(128, dim)
        self.A1 = SanokaLayer(dim, mul)
        
        self.B1 = SanokaLayer(dim, mul)
        self.C1 = SanokaLayer(dim, mul)
        
        self.D1 = SanokaLayer(dim, mul)
        self.E1 = SanokaLayer(dim, mul)
        self.F1 = SanokaLayer(dim, mul)
        
        self.A2 = SanokaLayer(dim, mul)
        self.B2 = SanokaLayer(dim, mul)
        self.C2 = SanokaLayer(dim, mul)
        self.D2 = SanokaLayer(dim, mul)
        self.E2 = SanokaLayer(dim, mul)
        self.F2 = SanokaLayer(dim, mul)
        
    def reset(self):
        self.A1.reset()
        
        self.B1.reset()
        self.C1.reset()
        
        self.D1.reset()
        self.E1.reset()
        self.F1.reset()
        
        self.A2.reset()
        self.B2.reset()
        self.C2.reset()
        self.D2.reset()
        self.E2.reset()
        self.F2.reset()
        
    def forward(self, x):
        if (self.Top):
            x = self.I(x)
        x = self.A1(x)
        
        x = self.B1(x)
        x = self.C1(x)
        
        x = self.D1(x)
        x = self.E1(x)
        x = self.F1(x)
        
        x = self.A2(x)
        x = self.B2(x)
        x = self.C2(x)
        x = self.D2(x)
        x = self.E2(x)
        x = self.F2(x)
        
        return x

class SanokaModel24(torch.nn.Module):
    def __init__(self, dim, mul=1):
        super().__init__()
        self.I = torch.nn.Linear(128, dim)
        self.A1 = SanokaLayer(dim, mul)
        
        self.B1 = SanokaLayer(dim, mul)
        self.C1 = SanokaLayer(dim, mul)
        
        self.D1 = SanokaLayer(dim, mul)
        self.E1 = SanokaLayer(dim, mul)
        self.F1 = SanokaLayer(dim, mul)
        
        self.A2 = SanokaLayer(dim, mul)
        self.B2 = SanokaLayer(dim, mul)
        self.C2 = SanokaLayer(dim, mul)
        self.D2 = SanokaLayer(dim, mul)
        self.E2 = SanokaLayer(dim, mul)
        self.F2 = SanokaLayer(dim, mul)
        
        self.A3 = SanokaLayer(dim, mul)
        self.B3 = SanokaLayer(dim, mul)
        self.C3 = SanokaLayer(dim, mul)
        self.D3 = SanokaLayer(dim, mul)
        self.E3 = SanokaLayer(dim, mul)
        self.F3 = SanokaLayer(dim, mul)
        
        self.A4 = SanokaLayer(dim, mul)
        self.B4 = SanokaLayer(dim, mul)
        self.C4 = SanokaLayer(dim, mul)
        self.D4 = SanokaLayer(dim, mul)
        self.E4 = SanokaLayer(dim, mul)
        self.F4 = SanokaLayer(dim, mul)
        
    def reset(self):
        self.A1.reset()
        
        self.B1.reset()
        self.C1.reset()
        
        self.D1.reset()
        self.E1.reset()
        self.F1.reset()
        
        self.A2.reset()
        self.B2.reset()
        self.C2.reset()
        self.D2.reset()
        self.E2.reset()
        self.F2.reset()
        
        self.A3.reset()
        self.B3.reset()
        self.C3.reset()
        self.D3.reset()
        self.E3.reset()
        self.F3.reset()
        
        self.A4.reset()
        self.B4.reset()
        self.C4.reset()
        self.D4.reset()
        self.E4.reset()
        self.F4.reset()
        
    def forward(self, x):
        x = self.I(x)
        x = self.A1(x)
        
        x = self.B1(x)
        x = self.C1(x)
        
        x = self.D1(x)
        x = self.E1(x)
        x = self.F1(x)
        
        x = self.A2(x)
        x = self.B2(x)
        x = self.C2(x)
        x = self.D2(x)
        x = self.E2(x)
        x = self.F2(x)
        
        x = self.A3(x)
        x = self.B3(x)
        x = self.C3(x)
        x = self.D3(x)
        x = self.E3(x)
        x = self.F3(x)
        
        x = self.A4(x)
        x = self.B4(x)
        x = self.C4(x)
        x = self.D4(x)
        x = self.E4(x)
        x = self.F4(x)
        return x

class OutputLayer (torch.nn.Module):
    def __init__(self, hiddendim, worddim=59000, heads=4):
        super().__init__()
        self.H = torch.nn.Linear(hiddendim, worddim)
    def forward(self, inpute):
        x = inpute
        x = self.H(x)
        return x       

def GOILOAD():
    fuf = open("table.txt", "r", encoding="UTF-8")
    goi = fuf.read().split("\n")
    fuf.close()
    chardim = len(goi[1:])
    charid = {goi[i+1].split()[0]:i for i in range(chardim-1)}
    return charid, [goi[ia+1].split()[0] for ia in range(chardim-1)]

datas = []
trues = []
lens = []
dones = 0
def Convert(buns, table):
    buns = buns.split("\n")
    tagger = MeCab.Tagger("-Owakati")
    w2v = Word2Vec.load("word2vec.model")
    data = []
    true = []
    lena = []
    for datac in range(len(buns)):
        #print(datac)
        #print(buns[datac])
        error = False
        try:
            buna = tagger.parse(buns[datac]).split()
            a = torch.from_numpy(w2v.wv[buna])
            b = torch.tensor([table[buna[ii]] for ii in range(len(buna))])
            ll = len(buna)
            c = ll
        except:
            print("ERROR")
        else:
            data.append(a)
            true.append(b)
            lena.append(c)
            print(datac)
    f = open("Train_Data.bin", "wb")
    pickle.dump((data, true, lena), f)
    f.close()
    return

def W2VMake(filepath="train_data.txt", mincount=70, worker=10):
    tagger = MeCab.Tagger("-Owakati")
    f = open(filepath, mode="r", encoding="UTF-8")
    texts = f.read().split("\n")
    f.close()
    dat = []
    print(len(texts))
    for a in range(len(texts)):
        dat.append(tagger.parse(texts[a]).split())
        print(a)
    model = Word2Vec(sentences=dat, vector_size=128, window=100, min_count=mincount, workers=worker)
    model.save("word2vec.model")
    model.wv.save_word2vec_format('table.txt')

def DataMake(filepath="train_data.txt", maxlen=129):
    table, i2w = GOILOAD()
    print(len(table))
    time.sleep(1)
    f = open(filepath, mode="r", encoding="UTF-8")
    txt = f.read()
    f.close()
    Convert(txt, table)
    return None

def PreTrain(Load=False, dim=512, outputdim=40000, lr=1e-04, epoch=10, epochload=1000,usedata=100000, onestep=100, uselen=64):
    global datas
    global trues
    global lens
    torch.manual_seed(1293431)
    #torch.manual_seed(576765)
    device1 = torch.device("cuda:1")
    device2 = torch.device("cuda:0")
    lossf = torch.nn.CrossEntropyLoss()
    model1 = SanokaModel(dim, 2, True).to(torch.bfloat16).to(device1)
    model2 = SanokaModel(dim, 2, False).to(torch.bfloat16).to(device2)
    output = OutputLayer(dim, outputdim).to(torch.bfloat16).to(device2)
    
    if (Load):
        model1.load_state_dict(torch.load("LLM1.pth", map_location=device1))
        model2.load_state_dict(torch.load("LLM2.pth", map_location=device2))
        output.load_state_dict(torch.load("output.pth", map_location=device2))
    model1Optim = torch.optim.Adam(model1.parameters(), lr=lr)
    model2Optim = torch.optim.Adam(model2.parameters(), lr=lr)
    outputO = torch.optim.Adam(output.parameters(), lr=lr)
    f = open("Train_Data.bin", "rb")
    datas, trues, lens = pickle.load(f)
    f.close()
    train_x = torch.zeros((epochload, uselen, 128)).to(torch.bfloat16).to(device1)
    train_y = torch.full((epochload, uselen), outputdim - 1, dtype=torch.long).to(device2)
    table, i2w = GOILOAD()
    base = 0
    epoch = int(np.floor((len(datas) / epochload) * epoch))
    print("データ量", len(datas))
    for epochs in range(epoch):
        train_x = train_x.detach()
        train_y = train_y.detach()
        if (base < len(datas) - epochload*2):
            base += epochload
        else:
            base = 0
        if (base > usedata):
            base = 0
        for b in range(epochload):
            a = b + base
            leng = lens[a]
            if (leng > uselen):
                leng = uselen
                
            train_x[b, :datas[a].shape[0]] = datas[a].to(torch.bfloat16).to(device1)[:uselen]
            train_y[b, :trues[a].shape[0]] = trues[a].to(device2).to(torch.long)[:uselen]
        epls = 0.00
        timem = time.time()
        for steps in range(epochload//onestep):
            model1.reset()
            model2.reset()
            oa = ""
            model1Optim.zero_grad()
            model2Optim.zero_grad()
            outputO.zero_grad()
            loss = 0.00
            for b in range(uselen-1):
                out = model1(train_x[steps*onestep:steps*onestep+onestep, b])
                out = model2(out.to(device2))
                out = output(out)
                loss += lossf(out, train_y[steps*onestep:steps*onestep+onestep, b+1])
                epls += loss
                
                sfo = torch.nn.functional.softmax(out[0], dim=-1)
                wid = torch.argmax(sfo, dim=-1).item()
                try:
                    wd = i2w[wid]
                except:
                    oa = oa + "ERROR"
                else:
                    oa = oa + wd
                
            loss.backward()
            #print(b)
            model1Optim.step()
            model2Optim.step()
            outputO.step()
        print("出力サンプル> ", oa[:32])
        print("epoch", epochs,"Train_epoch_sum_loss", epls.item(), "time", time.time() - timem)
        if (epochs % 10 == 9):
            torch.save(model1.state_dict(), "LLM1.pth")
            torch.save(model2.state_dict(), "LLM2.pth")
            torch.save(output.state_dict(), "output.pth")  
def Fineturning(Load=False, dim=512, outputdim=40000, lr=1e-04, epoch=10000, epochload=1000, onestep=200, uselen=32):
    global datas
    global trues
    global lens
    torch.manual_seed(1293431)
    #torch.manual_seed(576765)
    device1 = torch.device("cuda:1")
    device2 = torch.device("cuda:0")
    lossf = torch.nn.CrossEntropyLoss()
    model1 = SanokaModel(dim, 2, True).to(torch.bfloat16).to(device1)
    model2 = SanokaModel(dim, 2, False).to(torch.bfloat16).to(device2)
    output = OutputLayer(dim, outputdim).to(torch.bfloat16).to(device2)
    model1.load_state_dict(torch.load("LLM1.pth", map_location=device1))
    model2.load_state_dict(torch.load("LLM2.pth", map_location=device2))
    output.load_state_dict(torch.load("output.pth", map_location=device2))
    model1Optim = torch.optim.Adam(model1.parameters(), lr=lr)
    model2Optim = torch.optim.Adam(model2.parameters(), lr=lr)
    outputO = torch.optim.Adam(output.parameters(), lr=lr)
    f = open("Train_Data.bin", "rb")
    datas, trues, lens = pickle.load(f)
    f.close()
    train_x = torch.zeros((epochload, uselen, 128)).to(torch.bfloat16).to(device1)
    train_y = torch.full((epochload, uselen), outputdim - 1, dtype=torch.long).to(device2)
    table, i2w = GOILOAD()
    base = 0
    epoch = int(np.floor((len(datas) / epochload) * epoch))
    #print(epoch)
    for epochs in range(epoch):
        train_x = train_x.detach()
        train_y = train_y.detach()
        if (base < len(datas) - epochload*2):
            base += epochload
        else:
            base = 0
        for b in range(epochload):
            a = b + base
            #print(a)
            leng = lens[a]
            if (leng > uselen):
                leng = uselen
                
            train_x[b, :datas[a].shape[0]] = datas[a].to(torch.bfloat16).to(device1)[:uselen]
            train_y[b, :trues[a].shape[0]] = trues[a].to(device2).to(torch.long)[:uselen]
        epls = 0.00
        timem = time.time()
        for steps in range(epochload//onestep):
            model1.reset()
            model2.reset()
            oa = ""
            loss = 0.00
            model1Optim.zero_grad()
            model2Optim.zero_grad()
            outputO.zero_grad()
            for b in range(uselen-1):
                with torch.no_grad():
                    out = model1(train_x[steps*onestep:steps*onestep+onestep, b])
                    out = model2(out.to(device2))
                out = output(out)
                loss += lossf(out, train_y[steps*onestep:steps*onestep+onestep, b+1])
                epls += loss.item()
                
                sfo = torch.nn.functional.softmax(out[0], dim=-1)
                wid = torch.argmax(sfo, dim=-1).item()
                try:
                    wd = i2w[wid]
                except:
                    oa = oa + "ERROR"
                else:
                    oa = oa + wd
            loss.backward()
            #modelOptim.step()
            outputO.step()
        print("出力サンプル> ", oa[:32])
        print("epoch", epochs,"Train_epoch_sum_loss", epls, "time", time.time() - timem)
        if (epochs % 10 == 9):
            torch.save(output.state_dict(), "fineturning.pth")
def Predict(dim=512, outputdim=40000, maxlen=32):
    torch.manual_seed(1293431)
    device1 = torch.device("cuda:1")
    device2 = torch.device("cuda:0")
    table, i2w = GOILOAD()
    tagger = MeCab.Tagger("-Owakati")
    w2v = Word2Vec.load("word2vec.model")
    model1 = SanokaModel(dim, 2, True).to(device1)
    model2 = SanokaModel(dim, 2, False).to(device2)
    output = OutputLayer(dim, outputdim).to(device2)
    model1.load_state_dict(torch.load("LLM1.pth", map_location=device1))
    model2.load_state_dict(torch.load("LLM2.pth", map_location=device2))
    output.load_state_dict(torch.load("fineturning.pth", map_location=device2))
    #output.load_state_dict(torch.load("output.pth", map_location=device2))
    while(1):
        dd = input("Q> ") + ","
        
        data = []
        buna = tagger.parse(dd).split()
        print(buna)
        for a in range(len(buna)):
            try:
                data.append(torch.from_numpy(w2v.wv[buna[a]]).view(1, 1, 128).to(device1))
            except KeyError:
                print("Not Found")
        dat = torch.cat(data, dim=1).to(device1)
        oa = ""
        with torch.no_grad():
            model1.reset()
            model2.reset()
            for a in range(dat.shape[1] - 1):
                out = model1(dat[:, a])
                out = model2(out.to(device2))
            for b in range(maxlen - dat.shape[1]):
                out = model1(dat[:, -1])
                out = model2(out.to(device2))
                out = output(out)
                sfo = torch.nn.functional.softmax(out, dim=-1)
                wid = torch.argmax(sfo, dim=-1).item()
                if (wid != outputdim - 1):
                    try:
                        wd = i2w[wid]
                    except:
                        oa = oa + "ERROR"
                    else:
                        oa = oa + wd
                        dat = torch.cat([dat, torch.from_numpy(w2v.wv[wd]).to(device1).view(1, 1, 128)], dim=1)
        print("A> ", oa)
def ValidationLoss(dim=512, outputdim=40000, maxlen=32):
    torch.manual_seed(1293431)
    device1 = torch.device("cuda:1")
    device2 = torch.device("cuda:0")
    table, i2w = GOILOAD()
    tagger = MeCab.Tagger("-Owakati")
    w2v = Word2Vec.load("word2vec.model")
    model1 = SanokaModel(dim, 2, True).to(torch.bfloat16).to(device1)
    model2 = SanokaModel(dim, 2, False).to(torch.bfloat16).to(device2)
    output = OutputLayer(dim, outputdim).to(torch.bfloat16).to(device2)
    model1.load_state_dict(torch.load("LLM1.pth", map_location=device1))
    model2.load_state_dict(torch.load("LLM2.pth", map_location=device2))
    output.load_state_dict(torch.load("output.pth", map_location=device2))
    dd = input("TestData> ")
    lossf = torch.nn.CrossEntropyLoss()
    data = []
    buna = tagger.parse(dd).split()
    trued = torch.tensor([table[dfg] for dfg in buna]).to(torch.long).unsqueeze(dim=0)
    print(buna)
    print(trued)
    for a in range(len(buna)):
        try:
            data.append(torch.from_numpy(w2v.wv[buna[a]]).view(1, 1, 128).to(device1))
        except KeyError:
            print("Not Found")
    dat = torch.cat(data, dim=1).to(device1)
    oa = ""
    loss = 0.00
    with torch.no_grad():
        model1.reset()
        model2.reset()
        for a in range(dat.shape[1] - 1):
            out = model1(dat[:, a].to(torch.bfloat16))
            out = model2(out.to(device2))
            out = output(out)
            #print(out.shape)
            sfo = torch.nn.functional.softmax(out, dim=-1)
            wid = torch.argmax(sfo, dim=-1).item()
            try:
                wd = i2w[wid]
            except:
                oa = oa + "ERROR"
            else:
                oa = oa + wd
            loss += lossf(out, trued[:, a+1].to(device2))
    print("validationloss", loss.item() / dat.shape[1], "preview", oa)

if __name__ == "__main__": 
    #W2VMake()
    #DataMake()
    PreTrain(dim=1024, Load=False, outputdim=55000, lr=1e-03, onestep=75, uselen=64)
    #Fineturning(Load=False,dim=1024, outputdim=55000,lr=1e-03, onestep=500, uselen=64)
    #Predict(dim=1024, outputdim=55000, maxlen=48)
    #ValidationLoss(dim=1024, outputdim=55000, maxlen=64)
