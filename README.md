# NVIDIA-RAG

此次教學由 **NVIDIA** 兩位 senior solutions architect 教學，因為不確定是否可以公開就不放上 slides

RAG : 全名 Retrieval Augmented Generation 檢索增強生成，目的為無須額外訓練模型就能有相關資訊，最重要的是可以實時更新資料。

簡單來說， RAG 透過輸入的額外資訊切分成 chunk size ，並透過一 pretrained model 將其 embedding 並存於一 vector database，在使用者輸入問題時，此 vector database 會找出相關的資訊並送給 large language model，因此事實上他作用為多送一些你所問的問題的資訊給 model 
![image](https://github.com/david965154/NVIDIA-RAG/assets/145984683/508dded4-c84d-4708-9d09-ee13967f2735)

為了找出他所一同送入的資訊，我找了相關的函式
```python
question = "What is ChipNeMo?"
docs = vectorstore.similarity_search(question)
print(docs[0].page_content)

chipnemo : domain - adapted llms for chip design mingjie liu §, teodor - dumitru ene §, robert kirby §, chris cheng §, nathaniel pinckney §, rongjian liang § jonah alben, himyanshu anand, sanmitra banerjee, ismet bayraktaroglu, bonita bhaskaran, bryan catanzaro arjun chaudhuri, sharon clay, bill dally, laura dang, parikshit deshpande, siddhanth dhodhi, sameer halepete eric hill, jiashang hu, sumit jain, brucek khailany, george kokai, kishor kunal, xiaowei li charley lind, hao liu, stuart oberman, sujeet omar, sreedhar pratty, jonathan raiman, ambar sarkar zhengjiang shao, hanfei sun, pratik p suthar, varun tej, walker turner, kaizhe xu, haoxing ren nvidia abstract — chipnemo aims to explore the applications of large language models ( llms ) for industrial chip design. instead of directly deploying off - the - shelf commercial or open - source llms, we instead adopt the following domain adaptation techniques : custom tokenizers, domain - adaptive continued pretraining, su - pervised fine - tuning ( sft ) with domain - specific instructions, and domain - adapted retrieval models. we evaluate these methods on three selected llm applications for chip design : an engineering assistant chatbot, eda script generation, and bug summarization and analysis. our results show that these domain adaptation techniques enable significant llm performance improvements over general - purpose base models across the three evaluated applications, enabling up to 5x model size reduction with similar or better performance on
```
可以看到其找了 chunk size 大小最相關的資訊，這裡是論文的 abstract ，是最能代表整篇論文的部分，也透過這樣知道這樣搜尋相關資料並一同送入 llm 的方法是可行的。
