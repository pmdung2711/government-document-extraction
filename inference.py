from transformers import RobertaForSequenceClassification, RobertaConfig, AdamW
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import sys
from text_extraction import export_text
import cv2 
import torch
import os

class DocInfo:
    def __init__(self, texts, prediction_list):
        self.publisher = ""
        self.doc_code = ""
        self.date = ""
        self.summary = ""
        self.texts = texts
        print("Creating DocInfo object...")
        for i, text in enumerate(texts):
            if prediction_list[i] == 1:
                if self.publisher == '':
                    self.publisher += texts[i]
            if prediction_list[i] == 2:
                self.doc_code += (" " + texts[i])
            if prediction_list[i] == 3:
                self.date += (" " + texts[i])
            if prediction_list[i] == 4:
                self.summary += (" " + texts[i])
            

    def set_publisher(self, publisher):
        self.set_publisher = publisher

    def set_doc_code(self, doc_code):
        self.doc_code = doc_code

    def set_date(self, date):
        self.date = date


    def set_summary(self, summary):
        self.summary = summary 

    def set_texts(self, texts):
        self.texts = texts.copy()
    

class DocumentAnalysis:
    
    def __init__(self, model_path):

        #Load the pretrained PhoBERT Model
        print("Loading Classification...")
        self.config = RobertaConfig.from_pretrained(model_path + 'PhoBERT/config.json', from_tf=False, num_labels = 5, output_hidden_states=False,)
        self.phoBERT_cls = RobertaForSequenceClassification.from_pretrained(model_path + 'PhoBERT/model.bin',config=self.config)
        device = "cuda:0"
        self.phoBERT_cls = self.phoBERT_cls.to(device)
        self.phoBERT_cls.eval()
        print("Loading pre-trained model...")
        self.phoBERT_cls.load_state_dict(torch.load(model_path + 'roberta_state_dict_9bfb8319-01b2-4301-aa5a-756d390a98e1.pth'))
        print("Finished loading PhoBERT Classification model.")

        #Load the BPE and Vocabulary Dictionary
        print("Loading BPE and vocab dict ...")
        class BPE():
            bpe_codes = model_path + 'PhoBERT/bpe.codes'
        args = BPE()
        self.bpe = fastBPE(args)
        self.vocab = Dictionary()
        self.vocab.add_from_file(model_path + "PhoBERT/dict.txt")
        print("Finished loading BPE and vocab dict.")

        #Load the Text Recognizer 
        config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = 'weights/transformerocr.pth'
        config['cnn']['pretrained']=False
        config['device'] = 'cuda:0'
        config['predictor']['beamsearch']=False
        self.text_recognizer = Predictor(config)
        

    def prepare_features(self, seq):

        #Set maximum length for a sequence
        MAX_LEN = 256
        test_sents = [seq]

        #Convert text to ids
        test_ids = []
        for sent in test_sents:
            subwords = '<s>' + self.bpe.encode(sent) + ' </s>'
            encoded_sent = self.vocab.encode_line(subwords, append_eos=True, add_if_not_exist=False).long().tolist()
            test_ids.append(encoded_sent)
        

        
        #Create ids mask
        test_masks = []
        for sent in test_ids:
            mask = [int(token_id>0) for token_id in sent]
            test_masks.append(mask)

        return torch.tensor(test_ids).unsqueeze(0)[0], test_masks[0]



    def get_prediction(self, msg):
        self.phoBERT_cls.eval()
        input_msg, _ = self.prepare_features(msg)
        if torch.cuda.is_available():
            input_msg = input_msg.cuda()
        output = self.phoBERT_cls(input_msg)[0]
        _, pred_label = torch.max(output.data, 1)
        return pred_label 
    
    def get_prediction_test(self, msg):
        self.phoBERT_cls.eval()
        input_msg, _ = self.prepare_features(msg)
        if torch.cuda.is_available():
            input_msg = input_msg.cuda()
        output = self.phoBERT_cls(input_msg)[0]
        _, pred_label = torch.max(output.data, 1)
        prediction=list(["Văn bản thông thường", "Nhà xuất bản", "Số hiệu văn bản", "Ngày ban hành", "Tóm tắt nội dung"])[pred_label]
        return prediction 

    def get_prediction_list(self, texts):
        prediction_list = []
        #Get prediction for each sequence in the list
        for text in texts:
            prediction_list.append(self.get_prediction(text))

        return prediction_list

    def extract_information(self, image):
        #Create texts list
        texts = export_text(image, self.text_recognizer)
        prediction_list = self.get_prediction_list(texts)
        
        print("Return the prediction list")
        return DocInfo(texts, prediction_list)






if __name__ == "__main__":
    
    model_path = "weights/"
    test_seq = "Chúc mừng năm mới"
    test_seq2 = "Chúc mừng năm mới 2021, kính chúc mọi người"
    document_cls = DocumentAnalysis(model_path)
    test_image_path="test_data/Công văn 641_UBND-NC PDF.pdf.jpg"
    image = cv2.imread(test_image_path)
    doc_info = document_cls.extract_information(image)