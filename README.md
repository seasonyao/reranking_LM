If using `conda`, you can get this to work as follows:

```
conda create -n rerankLM python=3.8
conda activate rerankLM
conda install pytorch torchvision torchaudio cudatoolkit -c pytorch
conda install -c conda-forge jupyterlab
conda install -c conda-forge ipywidgets
conda install -c conda-forge matplotlib
conda install -c conda-forge jupyter_nbextensions_configurator
conda install -c conda-forge spacy
python -m spacy download en_core_web_sm
conda install pandas
conda install nltk

git clone https://github.com/seasonyao/reranking_LM.git
cd reranking_LM

git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```


