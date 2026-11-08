"""this is my scratch implementattion from assignmentt 3 of BPE tokeniser."""
import re
import json
from collections import Counter

PAT=re.compile(r" ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+")
class BPETokeniser:#defining the class
    def __init__(self):
        self.merges={}
        self.vocab={i: bytes([i]) for i in range(256)}
        self.cache={}

    @property
    def vocab_size(self):#vocab size
        return len(self.vocab)

    def _count_pairs(self,words,counts):#how actuall work we start counting how many words have same pair 
        pair_counts={}
        for word,count in zip(words,counts):
            for a,b in zip(word,word[1:]):
                pair_counts[(a,b)]=pair_counts.get((a,b),0)+count
        return pair_counts
    def _merge(self,ids,pair,new_id):
        out=[]
        i=0
        while i<len(ids):
            if i+1<len(ids) and ids[i]==pair[0] and ids[i+1]==pair[1]:
                out.append(new_id)
                i+=2
            else:
                out.append(ids[i])
                i+=1
        return out
    def train_slow(self,text,vocab_size):   # Version 1: recounts every pair each round (reference)
        chunks=PAT.findall(text)
        chunk_counts=Counter(chunks)
        words=[]
        counts=[]
        for chunk,n in chunk_counts.items():
            words.append(list(chunk.encode("utf-8")))
            counts.append(n)
        for r in range(vocab_size-256):
                pair_counts=self._count_pairs(words,counts)
                if not pair_counts:
                    break
                best = min(pair_counts, key=lambda p: (-pair_counts[p], p))
                new_id=256+r
                self.merges[best]=new_id
                self.vocab[new_id]=self.vocab[best[0]] + self.vocab[best[1]]
                words=[self._merge(w,best,new_id) for w in words]
    def train(self,text,vocab_size):        # Version 2: counts once, updates only affected words
        chunks=PAT.findall(text)
        chunk_counts=Counter(chunks)
        words=[]
        counts=[]
        for chunk,n in chunk_counts.items():
            words.append(list(chunk.encode("utf-8")))
            counts.append(n)
        pair_counts={}
        where={}
        for wi,(w,c) in enumerate(zip(words,counts)):
            for a,b in zip(w,w[1:]):
                pair_counts[(a,b)]=pair_counts.get((a,b),0)+c
                where.setdefault((a,b),set()).add(wi)
        for r in range(vocab_size-256):
            if not pair_counts:
                break
            best=min(pair_counts,key=lambda p:(-pair_counts[p],p))
            new_id=256+r
            self.merges[best]=new_id
            self.vocab[new_id]=self.vocab[best[0]]+self.vocab[best[1]]
            for wi in list(where[best]):
                w,c=words[wi],counts[wi]
                for a,b in zip(w,w[1:]):
                    pair_counts[(a,b)]-=c
                    if pair_counts[(a,b)]==0:
                        del pair_counts[(a,b)]
                    where[(a,b)].discard(wi)
                w=self._merge(w,best,new_id)
                words[wi]=w
                for a,b in zip(w,w[1:]):
                    pair_counts[(a,b)]=pair_counts.get((a,b),0)+c
                    where.setdefault((a,b),set()).add(wi)
            where.pop(best,None)
    def save(self,path):
        json.dump([[a,b,i] for (a,b),i in self.merges.items()],open(path,"w"))
    def load(self,path):
        for a,b,i in json.load(open(path)):
            self.merges[(a,b)]=i
            self.vocab[i]=self.vocab[a]+self.vocab[b]
    def _encode_chunk(self,chunk):
        if chunk in self.cache:
            return self.cache[chunk]
        ids=list(chunk.encode("utf-8"))
        while True:
            candidates=[p for p in zip(ids,ids[1:]) if p in self.merges]
            if not candidates:
                break
            best=min(candidates,key=lambda p:self.merges[p])
            ids=self._merge(ids,best,self.merges[best])
        self.cache[chunk]=ids
        return ids
    def encode(self,text):
        ids=[]
        for chunk in PAT.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids
    def decode(self,ids):
        data=b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8",errors="replace")


def train_tokenizer(train_text: str, vocab_size: int):
    tok=BPETokeniser()
    tok.train(train_text,vocab_size)
    return tok