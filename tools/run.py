import sys
import os
tpath=os.path.realpath(__file__).replace(u'/tools/run.py',u'')
# print(tpath)
sys.path.append(tpath)

from softmasked_bert.data.loaders.collator import DataCollatorForCsc
from pytorch_lightning.callbacks import ModelCheckpoint
from softmasked_bert.data.build import make_loaders
from softmasked_bert.data.loaders import get_csc_loader
from softmasked_bert.modeling.csc import SoftMaskedBertModel
from transformers import BertTokenizer
from bases import args_parse,train
from softmasked_bert.utils import get_abs_path
from softmasked_bert.data.processors.csc import preproc
import torch
from collections import OrderedDict

def train_mode():
    cfg = args_parse("csc/train_SoftMaskedBert.yml")
    if not os.path.exists(get_abs_path(cfg.DATASETS.TRAIN)):
        preproc()
    tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)
    collator = DataCollatorForCsc(tokenizer=tokenizer)
    if cfg.MODEL.NAME == "SoftMaskedBertModel":
        model = SoftMaskedBertModel(cfg, tokenizer)
    if len(cfg.MODEL.WEIGHTS) > 0:
        ckpt_path = get_abs_path(cfg.OUTPUT_DIR, cfg.MODEL.WEIGHTS)
        model.load_from_checkpoint(ckpt_path, cfg=cfg, tokenizer=tokenizer)
    loaders = make_loaders(cfg, get_csc_loader, _collate_fn=collator)
    ckpt_callback = ModelCheckpoint(monitor='val_loss',dirpath=get_abs_path(cfg.OUTPUT_DIR),filename='{epoch:02d}-{val_loss:.5f}',save_top_k=1,mode='min')
    train(cfg, model, loaders, ckpt_callback)
    return

def load_model(ckpt_file='SoftMaskedBert/epoch=06-val_loss=0.03013.ckpt', config_file='csc/train_SoftMaskedBert.yml'):
    # Example:
    # ckpt_fn = 'SoftMaskedBert/epoch=06-val_loss=0.03013.ckpt' (find in checkpoints)
    # config_file = 'csc/train_SoftMaskedBert.yml' (find in configs)
    from softmasked_bert.config import cfg

    cp = get_abs_path('checkpoints', ckpt_file)
    cfg.merge_from_file(get_abs_path('configs', config_file))
    tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)
    if cfg.MODEL.NAME == "SoftMaskedBertModel":
        model = SoftMaskedBertModel.load_from_checkpoint(cp,cfg=cfg,tokenizer=tokenizer)
    model.eval()
    model.to(cfg.MODEL.DEVICE)
    return model

def inference_mode(ckpt_file='SoftMaskedBert/epoch=06-val_loss=0.03013.ckpt', config_file='csc/train_SoftMaskedBert.yml',text_file='',text_sentence='',mode='0'):
    model = load_model(ckpt_file,config_file)
    texts = []
    if mode=='1':
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                texts.append(line.strip())
        f.close()
        corrected_texts = model.predict(texts)
        with open('result/result.txt','w',encoding='utf-8') as fr:
            for c in corrected_texts:
                fr.write(c+'\n')
        fr.close()
        i=0
        for i in range(len(texts)):
            print('第%s句错句：%s'%(str(i+1),texts[i]))
            print('第%s句纠正：%s'%(str(i+1),corrected_texts[i]))
        return corrected_texts
    elif mode == '2':
        # texts = input('错句：')
        texts.append(text_sentence)
        corrected_texts=model.predict(texts)
        print('纠正：%s'%corrected_texts[0])
        return corrected_texts
    else:
        print('模式选择错误，请在1或2中选择')
        return

def convert_mode(ckpt_file='epoch=06-val_loss=0.03013.ckpt', model_name='SoftMaskedBert'):
    file_dir = get_abs_path("checkpoints", model_name)
    state_dict = torch.load((os.path.join(file_dir, ckpt_file)))['state_dict']
    new_state_dict = OrderedDict()
    new_state_dict = state_dict
    torch.save(new_state_dict, os.path.join(file_dir, 'pytorch_model.bin'))
    return

def main():
    print('欢迎使用自动文本纠错应用,下面我将对本工具的使用进行一个简单的介绍：\n本工具主要分为训练、纠错（预测）和模型迁移三个部分\n\
    训练部分：\n\
        如果您有意向使用本工具的训练功能，请在下面的模式选择中输入1；\n\
    纠错部分：\n\
        如果您想要使用该模型进行预测请在模式选择中输入2；\n\
    模型迁移部分：\n\
        如果您仅仅需要将训练好的模型权重提取出来方便其他项目使用，请在模式选择中输入3；\n\
    当您希望退出该工具时，请在模式选择中输入0；\n祝您使用愉快，希望有帮到您。\n')

    mode = input('请输入您选择的模式：')
    if mode == '0':
        print('您已退出应用，期待您的下次使用')
        return
    elif mode == '1':
        print('将为您开始训练模型，这一阶段耗时较长，请耐心等待……')
        train_mode()
        print('久等了，模型训练完成')
        return
    elif mode == '2':
        flag = '0'
        print('将开始进行文本纠错……请稍后……')
        ckpt_file=input('请输入选择的ckpt模型，例如"SoftMaskedBert/epoch=06-val_loss=0.03013.ckpt"：')
        config_file=input('请输入选择的config文件，例如"csc/train_SoftMaskedBert.yml"：')
        print('请问您是希望将txt文件下的文本进行纠错还是直接输入文本进行纠错？前者选择1，后者选择2')
        flag = input('输入您的选择：')
        if flag == '1':
            text_file='datasets/test/test.txt'
            print('将直接在datasets/test/test.txt下进行纠错，纠错后的文本将直接输出至终端,然后保存至result/result.txt……')
            inference_mode(ckpt_file,config_file,text_file,'','1')
            print('纠错完成')
        elif flag == '2':
            print('将会根据您输入的每句话进行纠错……输入0时退出')
            while True:
                text_sentence=input('错句：')
                if text_sentence == '0':
                    print('成功退出逐句纠错')
                    break
                inference_mode(ckpt_file,config_file,'',text_sentence,'2')
            # print('纠错完成')
        return
    elif mode == '3':
        print('将为您导出模型的权重……')
        model_name=input('请输入选择的model，例如"SoftMaskedBert"：')
        ckpt_file=input('请输入选择的ckpt模型，例如"epoch=06-val_loss=0.03013.ckpt"：')
        convert_mode(ckpt_file,model_name)
        print('模型权重导出完成，您可以在checkpoints/%s下找到它'%model_name)
        return
    else:
        print('这不是一个有效的处理模式，请在0、1、2、3中进行选择！')
        return

if __name__=='__main__':
    main()
