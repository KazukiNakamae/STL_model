# STL_model

This is one of the fine-tuned models, named STL model, from [zhihan1996/DNABERT-2-117M
](https://huggingface.co/zhihan1996/DNABERT-2-117M).

The model data is available in Hugging Face: [https://huggingface.co/KazukiNakamae/STLmodel](https://huggingface.co/KazukiNakamae/STLmodel)

---

The STL model can predict the RNA offtarget induced by cytosine base editors (CBEs).

Here is an example of using the model for RNA-off-target prediction.

**pred_rna_offtarget.py:**

```python
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

__authors__ = ["Kazuki Nakamae"]
__version__ = "1.0.0"

def pred_rna_offtarget(dna, model_dir):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, trust_remote_code=True).to(device)
    except Exception as e:
        print(f"Error loading model from {model_dir}: {e}")
        sys.exit(1)

    inputs = tokenizer(dna, return_tensors='pt')
    model.eval() 
    with torch.no_grad():
      outputs = model(
          inputs["input_ids"].to(device), 
          inputs["attention_mask"].to(device),
        )
    print("[Negative, Positive]")
    print(outputs.logits)
    y_preds = np.argmax(outputs.logits.to('cpu').detach().numpy().copy(), axis=1)
    
    def id2label(x):
        return model.config.id2label[x]
    y_dash = [id2label(x) for x in y_preds]
    print("Result:")
    print(y_dash)
    # LABEL_0: Not RNA-offtarget / LABEL_1: RNA-offtarget
    return (dna, y_dash)

def print_usage():
    print(f"Usage: {sys.argv[0]} <input DNA sequence> <DNABERT-2 model directory>")
    print("Options:")
    print("  -h, --help    Show this help message and exit")
    print("  -v, --version Show version information and exit")

def print_version():
    print(f"{sys.argv[0]} version {__version__}")
    print("Authors:", ", ".join(__authors__))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help"):
            print_usage()
            sys.exit(0)
        elif len(sys.argv) == 2 and sys.argv[1] in ("-v", "--version"):
            print_version()
            sys.exit(0)
        else:
            print_usage()
            sys.exit(1)
    
    dna = sys.argv[1]
    model_dir = sys.argv[2]

    pred_rna_offtarget(dna, model_dir)
```

```bash
$ python pred_rna_offtarget.py GGCAGGGCTGGGGAAGCTTACTGTGTCCAAGAGCCTGCTG KazukiNakamae/STLmodel;
[Negative, Positive]
tensor([[-1.6383,  1.4502]])
Result:
['LABEL_1']
$ python pred_rna_offtarget.py GTCATCTAACAAAAATATTCCGTTGCAGGAAAAGCAAGCT KazukiNakamae/STLmodel;
[Negative, Positive]
tensor([[ 0.5446, -0.5105]])
Result:
['LABEL_0']
```

#### Developers of the fine-tuned model
- [Takayuki Suzuki](https://github.com/szktkyk)
