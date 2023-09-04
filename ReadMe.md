# Language Modelling based on transformer
## The following is the code for a decoder based language model, similar to the architecture principle of GPT
## The task of this language model is to predict the next word given the previous history

### Directory Structure

```
.
├── dataset.py
├── eval.py
├── model.py
├── ReadMe.md
└── train.py
└── decoder_model.pt
└── decoder_stats.txt

```

### Download the decoder_model according to the directory structure from:

#### Link: https://iiitaphyd-my.sharepoint.com/:u:/g/personal/advaith_malladi_research_iiit_ac_in/ERKC8U7EvdZCik-xp4paIooBPZaFiMGWXGPE3psuPFLY8Q?e=lFfmRq

### To check the arcitecture of my model and to check it's performance, open:

```
decoder_stats.txt
```

### To train the model, run:

```
python3 -W ignore train.py

```

### To evaluate the model, run:

```
python3 -W ignore eval.py

```
