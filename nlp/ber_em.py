from transformers import BertModel, BertTokenizer
import torch
class Encoder_Language():
    def __init__(self, BERT_PATH=r'../xinglin-data/text2len_v01/bert'):
        self.BERT_PATH = BERT_PATH

        # 加载 BertTokenizer 和 BertModel
        with torch.no_grad():
            self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
            self.bert_model = BertModel.from_pretrained(BERT_PATH)
        print('load bert model over')

    def get_vector(self, sentences):
        # sentences[bs, N]
        batch_size = len(sentences)
        max_length = 0
        encoded_sentences = []
        real_len = []
        # 编码每个并记录最大长度
        for sentence in sentences:
            inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            # 获取 BERT 嵌入
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state  # [n, sequence_length, bert_hidden_size]
            embeddings = embeddings.mean(dim=0, keepdim=True)#平均池化
            # print("embeddings",embeddings.size())
            real_len.append(embeddings.size(1))
            encoded_sentences.append(embeddings)
            max_length = max(max_length, embeddings.size(1))
       
        # 填充每个编码后的句子向量到相同的长度
        padded_encoded_sentences = []
        for sentence_encoding in encoded_sentences:
            padding = torch.zeros((sentence_encoding.size(0), max_length - sentence_encoding.size(1), 768), device=sentence_encoding.device)
            padded_sentence = torch.cat((sentence_encoding, padding), dim=1)
            # print("torch.tensor(padded_sentence)",padded_sentence.clone().detach().size())
            padded_encoded_sentences.append(torch.tensor(padded_sentence))

        # 生成掩码
        mask = torch.zeros(batch_size, max_length, dtype=torch.int)
        for i, sentence_encoding in enumerate(encoded_sentences):
            mask[i, :sentence_encoding.size(1)] = 1
        
        return padded_encoded_sentences, mask,real_len
# if __name__ == "__main__":

#     to_get_bert_embeddings = Encoder_Language()

#     # use batch_to_ids to convert sentences to character ids
#     # sentences = [['A', 'person', 'is', 'walking', 'forwards', 'and', 'waving', 'his', 'hand'],
#     #              ['A', 'human', 'is', 'walking', 'in', 'a', 'circle', 'A', 'human', 'is', 'walking', 'in', 'a', 'circle',],
#     #              ['A', 'person', 'is', 'playing', 'violin', 'while', 'singing']]

#     sentences = [['A person is walking forwards'],
#                  ['A person is walking forwards and waving his hand'],
#                  ['A person is walking ','A person is walking forwards his hand']]
#     # print(character_ids)

#     embeddings, mask = to_get_bert_embeddings.forward(sentences)

#     # print(embeddings)
#     print(embeddings)
#     print(mask)