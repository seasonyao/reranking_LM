If using `conda`, you can get this to work as follows:

```
git clone https://github.com/seasonyao/reranking_LM.git
cd reranking_LM
conda env create --name reranking_LM --file environment.yaml

git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```


