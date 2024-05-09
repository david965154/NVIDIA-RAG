# NVIDIA-RAG

此次教學由 **NVIDIA** 兩位 senior solutions architect 教學，因為不確定是否可以公開就不放上 slides

RAG : 全名 Retrieval Augmented Generation 檢索增強生成，目的為無須額外訓練模型就能有相關資訊，最重要的是可以實時更新資料。

簡單來說， RAG 透過輸入的額外資訊切分成 chunk size ，並透過一 pretrained model 將其 embedding 並存於一 vector database，在使用者輸入問題時，此 vector database 會找出相關的資訊並送給 large language model，因此事實上他作用為多送一些你所問的問題的資訊給 llm

![image](https://github.com/david965154/NVIDIA-RAG/assets/145984683/508dded4-c84d-4708-9d09-ee13967f2735)

為了找出他所一同送入的資訊，我找了相關的函式
```python
docs = vectorstore.similarity_search(question)
```

以下測試:

```python
question = "What is ChipNeMo?"
answer = llm.invoke(question)
```

原先輸出，此時模型並不知道 ChipNeMo 為何

```
ChipNeMo is not a widely known term or concept, and I was unable to find extensive information about it. The name ChipNeMo seems to be a combination of "Chip" and "Neuro-Morphic." Based on this, it might refer to a neuromorphic chip, which is a type of chip designed to mimic the structure and function of a human brain. However, without more context or specific information, I cannot provide a definitive answer. If you have more details about ChipNeMo, I would be happy to try to provide a more precise explanation.
```

透過尋找與 question 相關度最高的資訊後

```python
docs = vectorstore.similarity_search(question)
print(docs[0].page_content)
```

得到

```
chipnemo : domain - adapted llms for chip design mingjie liu §, teodor - dumitru ene §, robert kirby §, chris cheng §, nathaniel pinckney §, rongjian liang § jonah alben, himyanshu anand, sanmitra banerjee, ismet bayraktaroglu, bonita bhaskaran, bryan catanzaro arjun chaudhuri, sharon clay, bill dally, laura dang, parikshit deshpande, siddhanth dhodhi, sameer halepete eric hill, jiashang hu, sumit jain, brucek khailany, george kokai, kishor kunal, xiaowei li charley lind, hao liu, stuart oberman, sujeet omar, sreedhar pratty, jonathan raiman, ambar sarkar zhengjiang shao, hanfei sun, pratik p suthar, varun tej, walker turner, kaizhe xu, haoxing ren nvidia abstract — chipnemo aims to explore the applications of large language models ( llms ) for industrial chip design. instead of directly deploying off - the - shelf commercial or open - source llms, we instead adopt the following domain adaptation techniques : custom tokenizers, domain - adaptive continued pretraining, su - pervised fine - tuning ( sft ) with domain - specific instructions, and domain - adapted retrieval models. we evaluate these methods on three selected llm applications for chip design : an engineering assistant chatbot, eda script generation, and bug summarization and analysis. our results show that these domain adaptation techniques enable significant llm performance improvements over general - purpose base models across the three evaluated applications, enabling up to 5x model size reduction with similar or better performance on
```
可以看到其找了 chunk size 大小最相關的資訊，這裡是論文的 abstract ，是最能代表整篇論文的部分。現在將問題及上面的尋找出的資訊一同送入模型

```python
question = "chipnemo : domain - adapted llms for chip design mingjie liu §, teodor - dumitru ene §, robert kirby §, chris cheng §, nathaniel pinckney §, rongjian liang § jonah alben, himyanshu anand, sanmitra banerjee, ismet bayraktaroglu, bonita bhaskaran, bryan catanzaro arjun chaudhuri, sharon clay, bill dally, laura dang, parikshit deshpande, siddhanth dhodhi, sameer halepete eric hill, jiashang hu, sumit jain, brucek khailany, george kokai, kishor kunal, xiaowei li charley lind, hao liu, stuart oberman, sujeet omar, sreedhar pratty, jonathan raiman, ambar sarkar zhengjiang shao, hanfei sun, pratik p suthar, varun tej, walker turner, kaizhe xu, haoxing ren nvidia abstract — chipnemo aims to explore the applications of large language models ( llms ) for industrial chip design. instead of directly deploying off - the - shelf commercial or open - source llms, we instead adopt the following domain adaptation techniques : custom tokenizers, domain - adaptive continued pretraining, su - pervised fine - tuning ( sft ) with domain - specific instructions, and domain - adapted retrieval models. we evaluate these methods on three selected llm applications for chip design : an engineering assistant chatbot, eda script generation, and bug summarization and analysis. our results show that these domain adaptation techniques enable significant llm performance improvements over general - purpose base models across the three evaluated applications, enabling up to 5x model size reduction with similar or better performance on, What is ChipNeMo?"

answer = llm.invoke(question)
print(answer)
```

```
ChipNeMo is a project that aims to explore the applications of large language models (LLMs) in the field of industrial chip design. Instead of directly deploying off-the-shelf commercial or open-source LLMs, the project adopts domain adaptation techniques to better tailor the models for chip design.

The domain adaptation techniques used by ChipNeMo include custom tokenizers, domain-adaptive continued pretraining, supervised fine-tuning (SFT) with domain-specific instructions, and domain-adapted retrieval models. These techniques are evaluated on three selected LLM applications for chip design: an engineering assistant chatbot, electronic design automation (EDA) script generation, and bug summarization and analysis.

The results of the evaluation show that these domain adaptation techniques enable significant performance improvements in LLMs over general-purpose base models for the three evaluated applications. This enables up to a 5x model size reduction with similar or better performance, making the use of LLMs in chip design more practical and efficient.
```

若不以手動一同輸入，而是透過以下方是自動輸入

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever()    #Retriever is an interface that returns documents given an unstructured query
)
result = qa_chain.invoke({"query": question})
result['result']
```

則會得到

```
ChipNeMo is a project that explores the applications of Large Language Models (LLMs) for industrial chip design. Instead of directly deploying off-the-shelf LLMs, ChipNeMo adopts domain adaptation techniques such as custom tokenizers, domain-adaptive continued pretraining, supervised fine-tuning with domain-specific instructions, and domain-adapted retrieval models. These techniques are evaluated on three selected LLM applications for chip design: an engineering assistant chatbot, EDA script generation, and bug summarization and analysis. The goal is to enable significant LLM performance improvements over general-purpose base models, reducing model size while maintaining or improving performance on a range of design tasks.
```

具體比較兩回答 (使用 chatGPT 3.5)

```
Q : 1 和 2 哪個回答比較好

A : 兩個回答都提供了對ChipNeMo項目的描述，包括該項目使用的領域適應技術以及評估的應用。然而，第一個回答更加詳細和具體，提供了更多細節和量化結果，例如指出這些技術能夠實現多大的模型大小減小以及性能提升。因此，我會認為第一個回答更好，因為它提供了更深入的理解和評估。
```

可以看到手動輸入一串他找到相關度最高的資料反而會較佳，可能是因為其不單單只是添加找到的資訊，而是會額外做一些操作，雖然目前尚無法完全理解。不過透過這樣知道搜尋相關資料並一同送入 llm 而無須經過 fine-tune 的方法是可行的。
